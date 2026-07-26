"""Read-only scorer for the frozen B7 refmark screen (CARD.md § 4-5).

Prints the card's named quantities from `results/screen_<model>.json` —
every window number beside its width null AND the visible-evidence
floor (precondition 3). No cell computed here.

Run: .venv/bin/python -m experiments.explorations.task_hunt.refmark.score [model ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [16, 32]
CHANCE = 1 / 3


def acc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("acc_test")


def auc(cells, key):
    c = cells.get(key)
    return None if c is None else c.get("auc")


def fmt(x):
    return "  --  " if x is None else f"{x:+.3f}"


def score_model(key: str):
    z = json.loads((RES / f"screen_{key}.json").read_text())
    cells = z["cells"]
    rows = z["meta"]["rows"]
    print(f"\n{'=' * 72}\n{key}  (screen_hs {z['meta']['screen_hs']}; "
          f"user_echo {rows.get('user_echo')})\n{'=' * 72}")

    tokL = acc(cells, "rlam/tok_linear")
    tokM = acc(cells, "rlam/tok_mlp")
    floor = acc(cells, "rlam/position_floor")
    print(f"rlam: tok_lin {tokL:.3f}  tok_mlp {tokM:.3f}  "
          f"pos_floor {floor:.3f}  (chance {CHANCE:.3f})  "
          f"Q1 tok-floor = {tokL - floor:+.3f} (predicted < +0.10)")
    print("  T    vis_floor  ax_lin(abs)  g_ax_lin  width_lin  "
          "ax-vis | g_ax_mlp width_mlp")
    for T in AX_TS:
        vis = acc(cells, f"rlam/T{T}/visible_evidence_floor")
        ax = acc(cells, f"rlam/T{T}/actxmean_linear")
        axf = acc(cells, f"rlam/T{T}/actxmean_foreign_linear")
        axm = acc(cells, f"rlam/T{T}/actxmean_mlp")
        axfm = acc(cells, f"rlam/T{T}/actxmean_foreign_mlp")
        g = None if ax is None else ax - tokL
        w = None if ax is None or axf is None else ax - axf
        av = None if ax is None or vis is None else ax - vis
        gm = None if axm is None else axm - tokM
        wm = None if axm is None or axfm is None else axm - axfm
        print(f"  {T:>3}  {vis:.3f}      {ax:.3f}      {fmt(g)}   "
              f"{fmt(w)}    {fmt(av)} | {fmt(gm)}  {fmt(wm)}")
    print("  T    sc(=win-shuf)  wc(=win-foreign)  win-vis | "
          "mlp: sc     wc")
    for T in ORD_TS:
        wl = acc(cells, f"rlam/T{T}/win_linear")
        sh = acc(cells, f"rlam/T{T}/win_shuf_linear")
        fo = acc(cells, f"rlam/T{T}/win_foreign_linear")
        vis = acc(cells, f"rlam/T{T}/visible_evidence_floor")
        sc = None if wl is None or sh is None else wl - sh
        wc = None if wl is None or fo is None else wl - fo
        wv = None if wl is None or vis is None else wl - vis
        wlm = acc(cells, f"rlam/T{T}/win_mlp")
        shm = acc(cells, f"rlam/T{T}/win_shuf_mlp")
        fom = acc(cells, f"rlam/T{T}/win_foreign_mlp")
        scm = None if wlm is None or shm is None else wlm - shm
        wcm = None if wlm is None or fom is None else wlm - fom
        print(f"  {T:>3}  {fmt(sc)}        {fmt(wc)}          "
              f"{fmt(wv)} | {fmt(scm)} {fmt(wcm)}")
    nw = acc(cells, "rlam/T16/null_win_linear")
    nt = acc(cells, "rlam/null_tok_linear")
    if nw is not None:
        print(f"  nulls T16: win {nw:.3f} tok {nt:.3f}")

    a_tok = auc(cells, "anchor/tok_linear")
    a_ax = auc(cells, "anchor/T16/actxmean_linear")
    if a_tok is not None:
        print(f"anchor is_marker (AUC): tok {a_tok:.3f}  "
              f"T16 actxmean {fmt(a_ax)} (calibration only)")

    wtl = auc(cells, "wd/tok_linear")
    if wtl is not None:
        print(f"wd (binary AUC, {rows.get('wd/test', {})}): "
              f"tok_lin {wtl:.3f}  tok_mlp {fmt(auc(cells, 'wd/tok_mlp'))}")
        for T in (16, 32, 64):
            a = auc(cells, f"wd/T{T}/actxmean_linear")
            af = auc(cells, f"wd/T{T}/actxmean_foreign_linear")
            am = auc(cells, f"wd/T{T}/actxmean_mlp")
            wv = auc(cells, f"wd/T{T}/win_linear")
            sv = auc(cells, f"wd/T{T}/win_shuf_linear")
            g = None if a is None else a - wtl
            w = None if a is None or af is None else a - af
            sc = None if wv is None or sv is None else wv - sv
            print(f"  T{T}: ax-tok {fmt(g)}  ax-foreign {fmt(w)}  "
                  f"ax_mlp {fmt(am)}  wd_sc {fmt(sc)}")
    else:
        print("wd: NO CELLS (blocks any KEEP, card § 5)")


def main():
    for k in (sys.argv[1:] or ["gpt2", "llama31_8b"]):
        if (RES / f"screen_{k}.json").exists():
            score_model(k)
        else:
            print(f"[{k}] no results file yet")


if __name__ == "__main__":
    main()
