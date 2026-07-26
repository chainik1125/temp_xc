"""Read-only scorer for the frozen B8 slen screen (CARD.md § 4-6).

Computes the card's named quantities from `results/screen_<model>.json`
and prints the P1-P5 / KEEP-KILL / ladder evaluation mechanically — no
cell is computed here, no number invented; every line traces to cells
in the results file. Verdict language stays with the LOG entry.

Run: .venv/bin/python -m experiments.explorations.task_hunt.slen.score [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FACES = ("lat", "lev", "disp")
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [4, 8, 16, 32]
QUOTE_TS = [16, 32]                    # card § 5 P3 scoring Ts
CHANCE = 1 / 3


def acc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("acc_test")


def auc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("auc")


def fmt(x):
    return "  --  " if x is None else f"{x:+.3f}" if isinstance(x, float) else x


def score_model(key: str):
    z = json.loads((RES / f"screen_{key}.json").read_text())
    cells = z["cells"]
    print(f"\n{'=' * 72}\n{key}  (screen_hs {z['meta']['screen_hs']}, "
          f"rows meta in file)\n{'=' * 72}")

    for face in FACES:
        tokL = acc(cells, f"{face}/tok_linear")
        tokM = acc(cells, f"{face}/tok_mlp")
        floor = acc(cells, f"{face}/position_floor")
        if tokL is None:
            print(f"\n-- {face}: NO CELLS --")
            continue
        print(f"\n-- {face} --  tok_lin {tokL:.3f}  tok_mlp {tokM:.3f}  "
              f"pos_floor {floor:.3f}  (chance {CHANCE:.3f})")
        print("   P1 tok_lin - floor = "
              f"{tokL - floor:+.3f}  (bar >= +0.05)")

        print("   T    g_ax_lin  width_lin | g_ax_mlp  width_mlp")
        for T in AX_TS:
            ax = acc(cells, f"{face}/T{T}/actxmean_linear")
            axf = acc(cells, f"{face}/T{T}/actxmean_foreign_linear")
            axm = acc(cells, f"{face}/T{T}/actxmean_mlp")
            axfm = acc(cells, f"{face}/T{T}/actxmean_foreign_mlp")
            g = None if ax is None else ax - tokL
            w = None if ax is None or axf is None else ax - axf
            gm = None if axm is None else axm - tokM
            wm = None if axm is None or axfm is None else axm - axfm
            print(f"   {T:>3}  {fmt(g)}   {fmt(w)}   |  {fmt(gm)}   {fmt(wm)}")

        print("   T    win_lin   sc(=win-shuf)  wc(=win-foreign)  "
              "sc/wc   | mlp: sc      wc")
        for T in ORD_TS:
            wl = acc(cells, f"{face}/T{T}/win_linear")
            sh = acc(cells, f"{face}/T{T}/win_shuf_linear")
            fo = acc(cells, f"{face}/T{T}/win_foreign_linear")
            sc = None if wl is None or sh is None else wl - sh
            wc = None if wl is None or fo is None else wl - fo
            frac = (None if sc is None or wc is None or wc <= 0
                    else sc / wc)
            wlm = acc(cells, f"{face}/T{T}/win_mlp")
            shm = acc(cells, f"{face}/T{T}/win_shuf_mlp")
            fom = acc(cells, f"{face}/T{T}/win_foreign_mlp")
            scm = None if wlm is None or shm is None else wlm - shm
            wcm = None if wlm is None or fom is None else wlm - fom
            print(f"   {T:>3}  {fmt(wl and wl - 0)}   {fmt(sc)}        "
                  f"{fmt(wc)}          {fmt(frac)} | {fmt(scm)}  {fmt(wcm)}")

        nw = acc(cells, f"{face}/T16/null_win_linear")
        nt = acc(cells, f"{face}/null_tok_linear")
        if nw is not None:
            print(f"   nulls T16: win {nw:.3f} tok {nt:.3f} "
                  f"(|null-chance| = {abs(nw - CHANCE):.3f}/"
                  f"{abs(nt - CHANCE):.3f})")

        wtl = auc(cells, f"{face}/wd/tok_linear")
        if wtl is not None:
            print(f"   wd (binary AUC): tok_lin {wtl:.3f}  "
                  f"tok_mlp {fmt(auc(cells, f'{face}/wd/tok_mlp'))}")
            for T in (16, 32, 64):
                a = auc(cells, f"{face}/wd/T{T}/actxmean_linear")
                af = auc(cells, f"{face}/wd/T{T}/actxmean_foreign_linear")
                am = auc(cells, f"{face}/wd/T{T}/actxmean_mlp")
                wv = auc(cells, f"{face}/wd/T{T}/win_linear")
                sv = auc(cells, f"{face}/wd/T{T}/win_shuf_linear")
                g = None if a is None else a - wtl
                w = None if a is None or af is None else a - af
                sc = None if wv is None or sv is None else wv - sv
                print(f"     T{T}: ax-tok {fmt(g)} ax-foreign {fmt(w)} "
                      f"ax_mlp {fmt(am)} wd_sc {fmt(sc)}")

    # ---- ladder table (card § 5 P3) ----
    print(f"\n-- LADDER (linear, T in {QUOTE_TS}) --")
    for T in QUOTE_TS:
        row = []
        for face in FACES:
            wl = acc(cells, f"{face}/T{T}/win_linear")
            sh = acc(cells, f"{face}/T{T}/win_shuf_linear")
            row.append(None if wl is None or sh is None else wl - sh)
        lat, lev, disp = row
        order = (lat is not None and lev is not None and disp is not None
                 and lat > lev > disp)
        print(f"   T{T}: sc_lat {fmt(lat)}  sc_lev {fmt(lev)}  "
              f"sc_disp {fmt(disp)}   lat>lev>disp: {order}")


def main():
    for k in (sys.argv[1:] or ["gpt2", "llama31_8b"]):
        if (RES / f"screen_{k}.json").exists():
            score_model(k)
        else:
            print(f"[{k}] no results file yet")


if __name__ == "__main__":
    main()
