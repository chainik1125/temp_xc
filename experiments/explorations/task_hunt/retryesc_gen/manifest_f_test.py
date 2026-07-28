"""Direct test of the `claim_zone` under-read — WHY the aim missed high.

`retryesc_gen` was aimed at `floor_excess` 0.185 using `claim_zone`'s
`frac_in_window[T64]` and came out at 0.261 measured. At 16:05 I offered a
hypothesis for the +0.076 gap — that `claim_zone` reads the RAW eligible
population while the floor is fit on the CLASS-BALANCED manifest — and
flagged it as needing a direct test before anyone relied on it. In the
19:11 beat I showed the additive constant is the wrong functional form
(it over-predicts both out-of-sample legs, with the bias declining
0.0755 -> 0.0686 -> 0.0656 as f rises), so curve-fitting the residual is a
dead end. This tests the MECHANISM instead.

**The standing lesson this operationalises:** check that your instrument
and your bar are computed on the SAME ROWS. The premeasure and the screen
do not use the same rows, and they never did:

    premeasure `elig` = (mask==0) & (probe_eligible==1) & (pos>=PRE_MIN_POS)
                        & isfinite(age)

    screen `elig`     = (mask==0) & (is_assistant==1) & (pos>=PRE_MIN_POS)
                        & (rows_flat>=0)          <- doc tails dropped
                        & (pos>=POS_MIN)          <- 64, not PRE_MIN_POS
                        & (pos % content>=OFF_MIN)<- back half of each chunk
                        & isfinite(age) & (bins>=0)
                        ... then position-stratified CLASS BALANCING, capped.

So there are five candidate mechanisms stacked, not one. This walks the
row population from the premeasure's to the screen's ONE FILTER AT A TIME
and reports `frac_in_window[T64]` at each step, so the gap is attributed
to whichever filter actually causes it rather than to the first plausible
story. Costs $0 and no GPU: the chunking geometry is deterministic from
the committed grids npz (`chunk_stream` reads the grid, not the pod
cache), so the manifest is reproducible locally after the pod is gone.

The target to explain, per leg (measured floor@T64 minus 1/3 chance):
    gpt2 0.2608 | gemma2_2b 0.2750 | llama31_8b 0.2886

Run: .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.manifest_f_test
Writes results/manifest_f_test.json
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels import wave3_lib as w3
from experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure import (
    section_age,
)
from experiments.explorations.task_hunt.labels.build_wave3_trio import (
    MIN_POS as PRE_MIN_POS,
    _terciles,
)
from experiments.explorations.task_hunt.labels.punctint_lib import (
    pos_strata,
    stratified_balanced_manifest,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    OFF_MIN,
    POS_MIN,
    _map_rows,
    _row_lookup,
)
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.retryesc_gen.cache_acts import TOK_TAG

HERE = Path(__file__).resolve().parent
GRIDS = HERE / "grids"
RES = HERE / "results"
T = 64
CHANCE = 1.0 / 3.0

# Measured floor accuracies at T64 from the committed screen artifacts.
# floor_excess = acc - 1/3 (3-class balanced manifest).
MEASURED_FLOOR = {"gpt2": 0.5941739082336426,
                  "gemma2_2b": 0.6083424687385559,
                  "llama31_8b": 0.6218662858009338}
# What I aimed with (claim_zone frac_in_window[T64], premeasure artifact).
AIMED = {"gpt2": 0.1853, "gemma2_2b": 0.2064, "llama31_8b": 0.2230}


def chunk_geometry(flat, off, key):
    """`cache_acts.chunk_stream` doc_idx WITHOUT needing a tokenizer or the
    pod cache: bos only changes token VALUES, never the chunk geometry."""
    n_prefix = 1 if MODELS[key]["bos"] else 0
    content = SEQ_LEN - n_prefix
    doc_idx = []
    for d in range(len(off) - 1):
        seg_len = off[d + 1] - off[d]
        for _ in range(0, seg_len - content + 1, content):
            doc_idx.append(d)
    return np.asarray(doc_idx, dtype=np.int32), n_prefix, content


def leg(key: str) -> dict:
    tag = TOK_TAG[key]
    z = np.load(GRIDS / f"elicit_retryesc_gen_v1_screen_{tag}.npz")
    off, first, mask = z["doc_off"], z["event_first"], z["event_mask"]
    is_assist, elig_f = z["is_assistant"], z["probe_eligible"]
    doc_split = z["doc_split"]
    n_docs = len(off) - 1

    age = np.concatenate([w3.sage_face(first[off[d]:off[d + 1]])
                          for d in range(n_docs)]).astype(np.float64)
    raw = np.concatenate([section_age(first[off[d]:off[d + 1]])
                          for d in range(n_docs)]).astype(np.float64)

    n_tok = len(raw)
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    train_rows = doc_split[doc_of] == 0

    doc_idx, n_prefix, content = chunk_geometry(z["token_ids"], off, key)
    lookup = _row_lookup(doc_idx)
    rows_flat, _ = _map_rows(doc_of, pos_of, lookup, content, n_prefix)

    def f_of(m):
        """frac_in_window over a row mask — claim_zone's own statistic."""
        a = raw[m & np.isfinite(raw)]
        return (float((a <= T).mean()), int(a.size)) if a.size else (float("nan"), 0)

    # ---- the ladder: premeasure population -> screen manifest ---------
    steps = []
    pre = (mask == 0) & (elig_f == 1) & (pos_of >= PRE_MIN_POS) & np.isfinite(age)
    steps.append(("0_premeasure_elig (claim_zone's own rows)", pre))

    s1 = (mask == 0) & (is_assist == 1) & (pos_of >= PRE_MIN_POS) & np.isfinite(age)
    steps.append(("1_swap probe_eligible -> is_assistant", s1))
    s2 = s1 & (rows_flat >= 0)
    steps.append(("2_+ drop document tails", s2))
    s3 = s2 & (pos_of >= POS_MIN)
    steps.append((f"3_+ pos >= POS_MIN ({POS_MIN})", s3))
    s4 = s3 & (pos_of % content >= OFF_MIN)
    steps.append((f"4_+ intra-chunk offset >= {OFF_MIN}", s4))

    # terciles are fit on the SCREEN's pre-population (screen.py order)
    elig_pre = (mask == 0) & (is_assist == 1) & (pos_of >= PRE_MIN_POS)
    bins, edges = _terciles(age, train_rows, elig_pre & np.isfinite(age))
    s5 = s4 & (bins >= 0)
    steps.append(("5_+ tercile-assigned (bins>=0)  == screen elig", s5))

    out_steps = [{"step": name, "f_T64": f_of(m)[0], "n_rows": f_of(m)[1]}
                 for name, m in steps]

    # ---- step 6: the class-balanced, position-stratified manifest -----
    # Reproduces screen.py's manifest exactly, including its seed.
    man = {}
    for split_name, flag in (("train", 0), ("test", 1)):
        m = s5 & (doc_split[doc_of] == flag)
        pool = np.flatnonzero(m)
        if not len(pool):
            continue
        strata = pos_strata(pos_of[pool], min_pos=POS_MIN)
        seed = zlib.crc32(f"retryesc_gen/{tag}/{split_name}".encode()) % 2 ** 16
        md, mp, mc = stratified_balanced_manifest(
            bins[pool], strata, doc_of[pool], pos_of[pool],
            cap=CAP[split_name], seed=seed)
        per = {int(c): int((mc == c).sum()) for c in (0, 1, 2)}
        if not per or min(per.values()) < MIN_ROWS:
            continue
        flat_idx = off[md] + mp
        a = raw[flat_idx]
        man[split_name] = {"f_T64": float((a <= T).mean()),
                           "n_rows": int(a.size),
                           "rows_per_class": per,
                           "_idx": flat_idx}

    # ---- step 7: what the floor ACTUALLY sees ------------------------
    # The refutation above forces a different question. `_FloorBank.feats`
    # is (sage_floor, dose_window_count) and dose_window_count is fed
    # `event_MASK`, not `event_first`. The mask spans the WHOLE event turn,
    # so a row whose event_first is older than T can still have masked
    # event tokens inside its trailing T-window. The floor's in-window
    # indicator is therefore P(any masked token in window), which is
    # STRICTLY LARGER than f = P(event_first in window) by roughly the
    # masked-turn width w -- an effective window of T + w, not T.
    dose = np.concatenate([w3.dose_window_count(mask[off[d]:off[d + 1]], T)
                           for d in range(n_docs)])
    seen = dose > 0
    for split_name, d in man.items():
        idx = d.pop("_idx")
        d["frac_masked_token_in_window"] = float(seen[idx].mean())
        d["floor_excess_predicted"] = float(seen[idx].mean())

    # measured masked-turn width, the proposed size of the bonus
    runs = []
    for dd in range(n_docs):
        m = mask[off[dd]:off[dd + 1]].astype(np.int64)
        if not m.any():
            continue
        edges_ = np.diff(np.concatenate([[0], m, [0]]))
        runs.extend((np.flatnonzero(edges_ == -1)
                     - np.flatnonzero(edges_ == 1)).tolist())
    w_med = float(np.median(runs)) if runs else float("nan")

    measured_fe = MEASURED_FLOOR[key] - CHANCE
    # The screen's floor is fit on train and SCORED on test.
    man_f = man.get("test", {}).get("f_T64", float("nan"))
    return {"leg": key, "tokenizer": tag,
            "masked_turn_width_median": w_med,
            "effective_window_T_plus_w": T + w_med,
            "tercile_edges": [float(v) for v in
                              (edges["edges"] if isinstance(edges, dict) else edges)],
            "aimed_f_claim_zone": AIMED[key],
            "measured_floor_acc_T64": MEASURED_FLOOR[key],
            "measured_floor_excess_T64": measured_fe,
            "ladder": out_steps,
            "manifest": man,
            "manifest_test_f_T64": man_f,
            "residual_manifest_vs_measured": man_f - measured_fe,
            "residual_claimzone_vs_measured": AIMED[key] - measured_fe}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    legs = [leg(k) for k in ("gpt2", "gemma2_2b", "llama31_8b")]

    for L in legs:
        print(f"\n=== {L['leg']} "
              f"(measured floor_excess {L['measured_floor_excess_T64']:.4f}) ===")
        for s in L["ladder"]:
            print(f"  {s['step']:<46} f={s['f_T64']:.4f}  n={s['n_rows']:,}")
        for sp, d in L["manifest"].items():
            print(f"  6_manifest[{sp}] (balanced+stratified){'':<11} "
                  f"f={d['f_T64']:.4f}  n={d['n_rows']:,}")
        print(f"  -> claim_zone residual  {L['residual_claimzone_vs_measured']:+.4f}"
              "   (row population is NOT the mechanism)")
        print(f"  -> manifest   residual  {L['residual_manifest_vs_measured']:+.4f}")
        print(f"  masked-turn width (median) w={L['masked_turn_width_median']:.0f}"
              f"  -> effective window T+w={L['effective_window_T_plus_w']:.0f}")
        for sp, d in L["manifest"].items():
            print(f"  7_P(masked token in window)[{sp}] "
                  f"= {d['frac_masked_token_in_window']:.4f}"
                  f"   vs measured floor_excess {L['measured_floor_excess_T64']:.4f}"
                  f"   resid {d['frac_masked_token_in_window'] - L['measured_floor_excess_T64']:+.4f}")

    (RES / "manifest_f_test.json").write_text(json.dumps(
        {"T": T, "chance": CHANCE, "legs": legs}, indent=2))
    print(f"\nwrote {RES / 'manifest_f_test.json'}")


if __name__ == "__main__":
    main()
