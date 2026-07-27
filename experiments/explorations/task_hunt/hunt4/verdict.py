"""hunt4/verdict.py — mechanical § 4 scoring of the gen-4 screen JSONs
(HUNT4_SCREEN_CARD.md; rules are the diafaces § 7 set verbatim, the
hunt3/bed236f1d existential form).

Per face × model:
  gain      = best window arm (actxmean/win, any probe, any T) − best tok arm
  null_ok   = that window arm − null_win_linear ≥ +0.02 (width null)
  floor_ok  = the arm beats visible_evidence_floor at ITS T
  wd_ok     = within-dialogue best window arm − wd best tok arm > 0
  KEEP iff an arm exists clearing all simultaneously (gain ≥ +0.05 ∧
  null ∧ its-own-T floor) ∧ wd_ok; KILL per the four clauses; else WEAK.
Order (Q3, table-routing only): wd win−shuf ≥ +0.03 at ANY
T ∈ {4,8,16,32} wherever the wd window gain is positive (ladder
extended per c1c5c949e — receipts at screen time).

Bundle flow (c1c5c949e): gpt2 + gemma2_2b screen first; llama31_8b
third leg runs for every face NOT already 2/2 KILL. Bundle verdict =
majority over PRESENT models; with only 2 present, 2/2 agreement is
the bundle verdict, splits stay PENDING-THIRD-LEG. Writes
results/verdict.json. PENDING TEAM REVIEW.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")
FACES = ("xnov", "tret", "sdom", "xtrend", "tretd")
AX_TS = (4, 8, 16, 32, 64)
ORD_TS = (4, 8, 16, 32)


def _acc(c, k):
    return c[k]["acc_test"] if k in c else None


def score_model(c, face):
    toks = [v for v in (_acc(c, f"{face}/tok_linear"),
                        _acc(c, f"{face}/tok_mlp")) if v is not None]
    if not toks:
        # screen SKIPped the face (MIN_ROWS starvation — e.g. tretd's
        # test class 0 under the position instruments): no cells to
        # score; contributes nothing to the bundle majority.
        return {"verdict": "SKIP",
                "reason": "no cells (screen MIN_ROWS skip)"}
    tok = max(toks)
    null_win = _acc(c, f"{face}/T16/null_win_linear")
    arms = {}
    for T in AX_TS:
        for arm in ("actxmean_linear", "actxmean_mlp", "win_linear",
                    "win_mlp"):
            v = _acc(c, f"{face}/T{T}/{arm}")
            if v is not None:
                arms[(T, arm)] = v
    # § 4 is EXISTENTIAL ("some matched-class window arm ..."): an arm
    # KEEP-qualifies iff IT clears gain ≥ +0.05 AND width null ≥ +0.02
    # AND ITS OWN T's visible floor, simultaneously.
    qual = {}
    for (T, arm), v in arms.items():
        fl = _acc(c, f"{face}/T{T}/visible_evidence_floor")
        if (v - tok >= 0.05 and null_win is not None
                and v - null_win >= 0.02 and fl is not None and v > fl):
            qual[(T, arm)] = v
    if qual:
        (bt, ba), bv = max(qual.items(), key=lambda kv: kv[1])
    else:
        (bt, ba), bv = max(arms.items(), key=lambda kv: kv[1])
    floor_at_bt = _acc(c, f"{face}/T{bt}/visible_evidence_floor")
    gain = bv - tok
    null_ok = null_win is not None and (bv - null_win) >= 0.02
    floor_ok = floor_at_bt is not None and bv > floor_at_bt
    exists_qualifying_arm = bool(qual)

    wd_tok_c = [v for v in (_acc(c, f"{face}_wd/tok_linear"),
                            _acc(c, f"{face}_wd/tok_mlp")) if v is not None]
    wd = {"present": bool(wd_tok_c)}
    if wd_tok_c:
        wd_tok = max(wd_tok_c)
        wd_arms = {}
        for T in (16, 32, 64):
            for arm in ("actxmean_linear", "actxmean_mlp", "win_linear"):
                v = _acc(c, f"{face}_wd/T{T}/{arm}")
                if v is not None:
                    wd_arms[(T, arm)] = v
        for T in (4, 8):
            v = _acc(c, f"{face}_wd/T{T}/win_linear")
            if v is not None:
                wd_arms[(T, "win_linear")] = v
        (wt, wa), wv = max(wd_arms.items(), key=lambda kv: kv[1])
        wd.update(tok=wd_tok, best=wv, best_arm=f"T{wt}/{wa}",
                  gain=round(wv - wd_tok, 4))
        order = {}
        for T in ORD_TS:
            w = _acc(c, f"{face}_wd/T{T}/win_linear")
            s = _acc(c, f"{face}_wd/T{T}/win_shuf_linear")
            if w is not None and s is not None:
                order[f"T{T}"] = round(w - s, 4)
        wd["order_margins"] = order
    wd_ok = wd.get("gain", -1) is not None and wd.get("gain", -1) > 0

    every_win = list(arms.values())
    kill1 = all(v - tok <= 0.02 for v in every_win)
    kill2 = null_win is not None and all(
        v - null_win < 0.02 for v in every_win)
    kill3 = all(
        (_acc(c, f"{face}/T{T}/visible_evidence_floor") or 0) >= v
        for (T, _a), v in arms.items())
    kill4 = wd.get("present") and not wd_ok
    keep = exists_qualifying_arm and wd_ok
    verdict = ("KEEP" if keep else
               "KILL" if (kill1 or kill2 or kill3 or kill4) else "WEAK")
    order_pass = any(m >= 0.03 for m in wd.get("order_margins", {}).values()) \
        if wd.get("gain", 0) and wd.get("gain", 0) > 0 else False
    return {
        "tok_best": round(tok, 4), "window_best": round(bv, 4),
        "window_best_arm": f"T{bt}/{ba}", "gain": round(gain, 4),
        "exists_qualifying_arm": exists_qualifying_arm,
        "null_win": null_win, "null_ok": bool(null_ok),
        "floor_at_best_T": floor_at_bt, "floor_ok": bool(floor_ok),
        "wd": wd, "wd_ok": bool(wd_ok),
        "kill_clauses": [k for k, f in
                         zip(("tok_within_002", "width_null", "floor",
                              "wd_erases"), (kill1, kill2, kill3, kill4))
                         if f],
        "order_pass_wd": bool(order_pass),
        "verdict": verdict,
    }


def main():
    out = {"card": "HUNT4_SCREEN_CARD.md § 4 (mechanical)",
           "status": "PENDING TEAM REVIEW", "models": {}, "bundle": {}}
    present = []
    for m in MODELS:
        p = HERE / "results" / f"screen_{m}.json"
        if not p.exists():
            out["models"][m] = "MISSING"
            continue
        present.append(m)
        c = json.loads(p.read_text())["cells"]
        out["models"][m] = {f: score_model(c, f) for f in FACES}
    for f in FACES:
        vs_all = {m: out["models"][m][f]["verdict"] for m in present}
        vs = [v for v in vs_all.values() if v != "SKIP"]
        keep_n, kill_n = vs.count("KEEP"), vs.count("KILL")
        if not vs:
            bundle = "SKIP-INFEASIBLE"
        elif len(vs) >= 3:
            bundle = ("KEEP" if keep_n >= 2 else
                      "KILL" if kill_n >= 2 else "WEAK")
        elif len(vs) == 2:
            bundle = vs[0] if vs[0] == vs[1] else "PENDING-THIRD-LEG"
        else:
            bundle = f"single-model {vs[0]} (others SKIP/absent)"
        orders = [out["models"][m][f].get("order_pass_wd", False)
                  for m in present]
        out["bundle"][f] = {
            "verdicts": vs_all,
            "bundle_verdict": bundle,
            "order_pass_models": int(sum(orders)),
            "third_leg_due": bool(len(present) == 2
                                  and not (len(vs) == 2
                                           and vs[0] == vs[1] == "KILL")),
            "table": ("panel-gate candidate" if bundle == "KEEP"
                      and sum(orders) >= 2 else
                      "breadth" if bundle == "KEEP" else "—")}
    dst = HERE / "results" / "verdict.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["bundle"], indent=1))
    print(f"[verdict] wrote {dst}")


if __name__ == "__main__":
    main()
