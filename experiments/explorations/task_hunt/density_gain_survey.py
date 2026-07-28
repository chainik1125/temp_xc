"""Does screen GAIN track IN-WINDOW EVENT MASS?  ($0, mac-c, 2026-07-28)

Receipt for the LOG entry of 14:07 London. Reads every committed
`*/results/screen_*.json` under `task_hunt/` and recomputes, per
face x model cell:

    tok        best per-token arm (linear or mlp)
    best       best window arm over T in {4,8,16,32,64}
    gain       best - tok                        <- the hunt4 s4 quantity
    floor      that T's `visible_evidence_floor`
    floor_excess = floor - chance                <- in-window event mass

`visible_evidence_floor` is fit on exactly two features --
`(censored_age, in_window_event_count)` (see `evalage/screen.py`
`_FloorBank.feats`). So `floor_excess` is not a proxy for the
information carried by events INSIDE the window: it is that
information, measured directly.

Result at time of writing: 150 cells / 48 artifacts, 49 cells and 15
distinct faces carry a floor. Pearson(floor_excess, gain) = +0.699 at
cell level, **+0.820 at face level** (Spearman +0.696 / +0.882).

CAVEAT, stated in the script because it is easy to lose: the band table
printed at the end uses edges (.05/.15/.25) chosen POST HOC, after
seeing these numbers. The correlation is the evidence; the bands are a
design target. Cells within a face are correlated, so the face-level
row (n=15) is the honest n, not n=49.

Run: .venv/bin/python -m experiments.explorations.task_hunt.density_gain_survey
"""
from __future__ import annotations

import collections
import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
AX_TS = (4, 8, 16, 32, 64)
WIN_ARMS = ("actxmean_linear", "actxmean_mlp", "win_linear")
BANDS = ((-1.0, .05, "< +0.05  (events ~never in window)"),
         (.05, .15, "+0.05..+0.15"),
         (.15, .25, "+0.15..+0.25"),
         (.25, 10., "> +0.25  (events dense in window)"))


def _pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db)


def _spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos
        return out
    return _pearson(rank(a), rank(b))


def collect() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(HERE / "*/results/screen_*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        cells = d.get("cells")
        if not isinstance(cells, dict):
            continue
        model = d.get("meta", {}).get("model", "?")
        faces = collections.defaultdict(dict)
        for k, v in cells.items():
            head, _, rest = k.partition("/")
            faces[head][rest] = v

        for face, c in faces.items():
            def acc(key):
                x = c.get(key)
                return x.get("acc_test") if isinstance(x, dict) else None

            tok = [a for a in (acc("tok_linear"), acc("tok_mlp"))
                   if a is not None]
            if not tok:
                continue
            probe = c.get("tok_linear") or c.get("tok_mlp") or {}
            k_cls = len(probe.get("per_class") or []) or None
            chance = 1.0 / k_cls if k_cls else None

            best = best_t = None
            for T in AX_TS:
                for arm in WIN_ARMS:
                    a = acc(f"T{T}/{arm}")
                    if a is not None and (best is None or a > best):
                        best, best_t = a, T
            if best is None:
                continue

            floor = acc(f"T{best_t}/visible_evidence_floor")
            rows.append(dict(
                file=Path(f).relative_to(HERE).as_posix(), model=model,
                face=face, k_cls=k_cls, chance=chance, tok=max(tok),
                best=best, gain=best - max(tok), best_t=best_t, floor=floor,
                floor_excess=(floor - chance)
                if (floor is not None and chance) else None))
    return rows


def main() -> None:
    rows = collect()
    have = [r for r in rows if r["floor_excess"] is not None]
    print(f"{len(rows)} face x model cells across "
          f"{len({r['file'] for r in rows})} screen artifacts; "
          f"{len(have)} carry a visible-evidence floor")

    fe = [r["floor_excess"] for r in have]
    gn = [r["gain"] for r in have]
    print(f"\nCELL LEVEL (n={len(have)})   pearson {_pearson(fe, gn):+.3f}"
          f"   spearman {_spearman(fe, gn):+.3f}")

    by_face = collections.defaultdict(list)
    for r in have:
        by_face[r["face"]].append(r)
    ff = [st.mean(x["floor_excess"] for x in xs) for xs in by_face.values()]
    gg = [st.mean(x["gain"] for x in xs) for xs in by_face.values()]
    print(f"FACE LEVEL (n={len(ff)})    pearson {_pearson(ff, gg):+.3f}"
          f"   spearman {_spearman(ff, gg):+.3f}   <- the honest n")

    print(f"\n{'face':<16}{'n':>3}{'floor_excess':>14}{'gain':>10}"
          f"{'margin best-floor':>20}")
    for face, xs in sorted(by_face.items(),
                           key=lambda kv: -st.mean(x["floor_excess"]
                                                   for x in kv[1])):
        print(f"{face:<16}{len(xs):>3}"
              f"{st.mean(x['floor_excess'] for x in xs):>+14.3f}"
              f"{st.mean(x['gain'] for x in xs):>+10.4f}"
              f"{st.mean(x['best'] - x['floor'] for x in xs):>+20.4f}")

    print("\n--- POST-HOC bands (design target, NOT independent evidence) ---")
    for lo, hi, label in BANDS:
        sel = [r for r in have if lo <= r["floor_excess"] < hi]
        if not sel:
            continue
        faces = [(f, xs) for f, xs in by_face.items()
                 if lo <= st.mean(x["floor_excess"] for x in xs) < hi]
        clear = [f for f, xs in faces
                 if all(x["gain"] >= 0.05 for x in xs)
                 and all(x["best"] > x["floor"] for x in xs)]
        print(f"{label:<38} cells={len(sel):>3}  "
              f"mean gain {st.mean(x['gain'] for x in sel):+.4f}  "
              f"gain>=+0.05 {sum(x['gain'] >= 0.05 for x in sel)}/{len(sel)}  "
              f"lost to own floor "
              f"{sum(x['best'] <= x['floor'] for x in sel)}/{len(sel)}  "
              f"faces clearing every cell {len(clear)}/{len(faces)} "
              f"{sorted(clear)}")


if __name__ == "__main__":
    main()
