"""AMPLIFIER TEST — does the window AMPLIFY a per-token signal, or GENERATE one?

$0, 0 pods. Reuses caches already on disk (evalage + retryesc_gen,
gemma2_2b L14 @512).

## The hypothesis under test (mine, from `7df9f25d8`)

Three corpora hinted that arm strength tracks tok strength:

    sycgen   tok +0.196   arm +0.308     KEEP 3/3
    evalage  tok +0.127   arm +0.197     WEAK
    retryesc tok +0.047   arm weak       KILL 3/3

If true, **the window amplifies an existing per-token signal rather than
generating one**, and the hunt's sourcing criterion is self-defeating:
`hunt-safety-gold-clew.md` demands **per-token-silent** tasks to suppress
`tok` and widen the gain — which would suppress the arm along with it.

Three points across three different screen configurations is not evidence.
**This runs many faces through ONE identical pipeline on ONE corpus**, so
the comparison is clean.

## ⚑ I WANT TO BE WRONG, AND SAY SO BEFORE THE NUMBERS

**A face with weak `tok` and strong `arm` is exactly what the hunt is
looking for.** If one exists, my amplifier framing dies and the program
gets a target. I am the author of the hypothesis and the runner of its
test, so the bias runs toward confirming it. Stating the preferred
refutation in advance is the guard.

## PRE-REGISTRATION — frozen before the run

- **A1 (the hypothesis).** `arm_excess` is monotone increasing in
  `tok_excess` across faces. Reported as Spearman ρ over the faces that
  build.
- **A2 (multiplicative form).** If amplification is multiplicative, the
  ratio `arm_excess / tok_excess` is roughly constant. Reported as the
  spread of that ratio; **no threshold is pre-set** — this is descriptive,
  and I will not invent a cutoff after seeing it.
- **⚑ A3 — THE KILL CONDITION, and the outcome I prefer.** If ANY face
  has `tok_excess <= 0.02` (per-token near-silent) **and**
  `arm_excess >= 0.05` (window clearly reads it), **A1 is REFUTED**, the
  amplifier framing is withdrawn, and **that face is the hunt's target**.
  One such face is enough; I am not requiring a pattern.
- **A4 (validity gate).** A face only counts if it beats its own
  `foreign` control and its `label_null` is near chance. A face that
  fails those is excluded and **listed as excluded**, not silently
  dropped.

**Falsifier against the test itself:** if fewer than 4 faces build on a
corpus, the Spearman is not reportable and I say so rather than quote a
correlation over 2–3 points.

Run:
  FACECMP_CACHE_ROOT=<scratch>/cache_evalage_512 PYTHONPATH=. \
    .venv/bin/python -m experiments.explorations.task_hunt.facecmp.amplifier_test
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CHANCE = 1.0 / 3.0
S = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-c-temp-xc/"
    "660a6cf4-2ce0-4db9-8201-37e38db3e1bf/scratchpad")

CORPORA = {
    "evalage": (f"{S}/cache_evalage_512",
                "experiments/explorations/task_hunt/evalage/grids",
                "elicit_evalage_screen_{tag}.npz"),
    "retryesc_gen": (f"{S}/cache_g2_512",
                     "experiments/explorations/task_hunt/retryesc_gen/grids",
                     "elicit_retryesc_gen_v1_screen_{tag}.npz"),
}


def spearman(xs, ys):
    """Rank correlation without scipy."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def run_corpus(name: str, key: str = "gemma2_2b"):
    import experiments.explorations.task_hunt.facecmp.arm_test as at
    from experiments.explorations.task_hunt.facecmp.face_battery import FACES

    root, grids, pat = CORPORA[name]
    at.CACHE_ROOT = Path(root)
    at.GRIDS = Path(grids)
    at.GRID_PAT = pat
    scratch = Path(__file__).resolve().parent / "results" / "_amp"
    scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    for face, fn, H in FACES:
        at.FACE, at.H, at.RES = face, H, scratch
        at.AX_TS, at.FOREIGN_TS = [16, 32, 64], [32, 64]
        at.rate_face = fn
        print(f"\n===== {name} / {face} (H={H}) =====", flush=True)
        try:
            at.screen(key)
        except Exception as e:
            print(f"  FAILED: {str(e)[:160]}")
            rows.append({"face": face, "excluded": f"build failed: {str(e)[:80]}"})
            continue
        p = scratch / f"arm_test_{key}.json"
        if not p.exists():
            rows.append({"face": face, "excluded": "no output"})
            continue
        d = json.loads(p.read_text())
        p.unlink(missing_ok=True)
        s = d.get("summary")
        if not s or "gain_vs_tok" not in s:
            rows.append({"face": face, "excluded": "insufficient rows"})
            continue
        c = d["cells"]
        foreign = (c.get(f"{face}/T64/actxmean_foreign_linear") or {}).get("acc_test")
        lnull = (c.get(f"{face}/T32/label_null") or {}).get("acc_test")
        rows.append({
            "face": face, "H": H,
            "tok": s["tok"], "arm": s["best_window"], "best_T": s["best_T"],
            "tok_excess": s["tok"] - CHANCE,
            "arm_excess": s["best_window"] - CHANCE,
            "gain": s["gain_vs_tok"],
            "foreign": foreign, "label_null": lnull,
            "valid": bool(foreign is not None and s["best_window"] > foreign
                          and lnull is not None and abs(lnull - CHANCE) < 0.05),
        })
    return rows


def main() -> None:
    which = os.environ.get("AMP_CORPORA", "evalage").split(",")
    out = {}
    for name in which:
        rows = run_corpus(name)
        out[name] = rows
        print(f"\n\n######## {name} ########")
        print(f"{'face':<16}{'tok_exc':>9}{'arm_exc':>9}{'gain':>9}"
              f"{'ratio':>8}{'T':>4}  valid")
        print("-" * 62)
        for r in rows:
            if "excluded" in r:
                print(f"{r['face']:<16}  EXCLUDED: {r['excluded']}")
                continue
            ratio = (r["arm_excess"] / r["tok_excess"]
                     if abs(r["tok_excess"]) > 1e-9 else float("nan"))
            print(f"{r['face']:<16}{r['tok_excess']:>+9.4f}{r['arm_excess']:>+9.4f}"
                  f"{r['gain']:>+9.4f}{ratio:>8.2f}{r['best_T']:>4}  {r['valid']}")

        ok = [r for r in rows if "excluded" not in r and r["valid"]]
        print(f"\n  faces built and valid: {len(ok)} / {len(rows)}")
        if len(ok) < 4:
            print("  ⚑ FEWER THAN 4 VALID FACES — Spearman NOT reportable "
                  "(pre-registered falsifier against the test itself).")
        else:
            rho = spearman([r["tok_excess"] for r in ok],
                           [r["arm_excess"] for r in ok])
            print(f"  A1 Spearman(tok_excess, arm_excess) = {rho:+.3f}")
            ratios = [r["arm_excess"] / r["tok_excess"] for r in ok
                      if abs(r["tok_excess"]) > 1e-9]
            if ratios:
                print(f"  A2 ratio arm/tok: min {min(ratios):.2f} "
                      f"max {max(ratios):.2f} spread {max(ratios)-min(ratios):.2f}")

        kill = [r for r in ok
                if r["tok_excess"] <= 0.02 and r["arm_excess"] >= 0.05]
        print(f"\n  ⚑ A3 KILL CONDITION (tok_excess<=0.02 AND arm_excess>=0.05): "
              f"{len(kill)} face(s)")
        if kill:
            for r in kill:
                print(f"     ** {r['face']}: tok {r['tok_excess']:+.4f} "
                      f"arm {r['arm_excess']:+.4f} — AMPLIFIER FRAMING REFUTED, "
                      f"and THIS IS THE HUNT'S TARGET **")
        else:
            print("     none — A1 survives on this corpus (NOT proven; "
                  "absence over 6 faces is weak evidence)")

    d = Path(__file__).resolve().parent / "results" / "amplifier"
    d.mkdir(parents=True, exist_ok=True)
    (d / "amplifier_test.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {d / 'amplifier_test.json'}")


if __name__ == "__main__":
    main()
