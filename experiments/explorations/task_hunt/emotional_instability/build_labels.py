"""Token-grid labels + probe manifests — emotional instability (CARD.md).

Consumes rollouts + judge outputs (results/scores.json, onset.json) and
produces flat-token-axis labels aligned with cache_acts.py's layout
(same deterministic chat-template tokenization; asserted against
acts/ids.npy when present):

  /workspace/emo_caches/labels.npz
     conv_of (N,) int16, turn_of (N,) int8 (assistant turn 0-7, -1
     elsewhere), score_of (N,) int8 (judged score of the containing
     response, -1 elsewhere), dist_onset (N,) int32 (onset_tok − p for
     convs with an onset; sentinel 10**6 otherwise), pos_in_turn (N,)
  results/manifests.npz  per task/split: rows = FLAT token indices, y
  results/label_stats.json

Tasks (matched per CARD; split by puzzle 80/20 rng(7); caps 4000/1500;
floor 300/class/split):
  ant4/ant8/ant16  binary anticipation ladder: pos offset∈[1,4]/[5,8]/
                   [9,16] before the conv's FIRST onset token; neg
                   assistant tokens ≥ 64 from any onset (pre-onset
                   regions + no-onset convs); guard (16,64) excluded;
                   matched on (token_id × assistant-turn index)
  esc3             3-class tercile of the CURRENT response's judged
                   score at pre-onset positions in the response;
                   matched on (turn index × pos-in-turn bucket)
  det              sanity anchor: post-onset (o, o+64] vs far-pre-onset
                   negatives, matched on turn index ONLY (the lexical
                   stamp IS the expected signal — identity matching
                   would erase what the anchor validates)

Eligibility everywhere: assistant tokens, flat-window depth ≥ 63 into
the conversation (T ≤ 64 windows stay in-conversation).

Run: .venv/bin/python -m experiments.explorations.task_hunt.emotional_instability.build_labels
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.replag.build_labels import (
    matched_sample,
)

MODEL_ID = "google/gemma-3-12b-it"
ROLLOUTS = Path("/workspace/emo_caches/rollouts")
ACTS = Path("/workspace/emo_caches/acts")
LAB = Path("/workspace/emo_caches/labels.npz")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MAX_LEN = 8192
P_MIN = 63
BANDS = {"ant4": (1, 4), "ant8": (5, 8), "ant16": (9, 16)}
GUARD_HI = 64                  # neg needs |dist| >= 64; guard (16,64)
DET_HI = 64
CAP = {"train": 4000, "test": 1500}
SPLIT_SEED = 7
MATCH_SEED = 1013
NO_ONSET = 10 ** 6
PIT_EDGES = [0, 64, 128, 256, 10 ** 9]     # pos-in-turn buckets


def build():
    from transformers import AutoTokenizer
    RES.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    scores = json.loads((RES / "scores.json").read_text())
    onsets = json.loads((RES / "onset.json").read_text())

    convs, order = {}, []
    for p in sorted(ROLLOUTS.glob("conv_*.json")):
        name = p.stem[5:]
        convs[name] = json.loads(p.read_text())
        order.append(name)

    acts_index = (json.loads((ACTS / "index.json").read_text())
                  if (ACTS / "index.json").exists() else None)
    acts_ids = (np.load(ACTS / "ids.npy", mmap_mode="r")
                if acts_index else None)

    N = (acts_index["n_tokens"] if acts_index else
         sum(len(tok.apply_chat_template(convs[n]["messages"],
                                         tokenize=True))
             for n in order))
    conv_of = np.full(N, -1, dtype=np.int16)
    turn_of = np.full(N, -1, dtype=np.int8)
    score_of = np.full(N, -1, dtype=np.int8)
    dist_onset = np.full(N, NO_ONSET, dtype=np.int32)
    pos_in_turn = np.full(N, -1, dtype=np.int32)
    ids_all = np.zeros(N, dtype=np.int32)
    conv_start = {}
    stats = {"onset_match_fail": [], "span_fail": [],
             "ids_mismatch": []}

    cursor = 0
    names = (sorted(order, key=lambda n: len(tok.apply_chat_template(
        convs[n]["messages"], tokenize=True))) if acts_index is None
        else sorted(order, key=lambda n: acts_index["convs"][n][0]))
    for ci, name in enumerate(names):
        msgs = convs[name]["messages"]
        ids = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=False)
        ids = ids[:MAX_LEN]
        n = len(ids)
        if acts_index:
            s, e = acts_index["convs"][name]
            assert (s, e) == (cursor, cursor + n), f"layout {name}"
            if not np.array_equal(np.asarray(ids, dtype=np.int32),
                                  acts_ids[s:e]):
                stats["ids_mismatch"].append(name)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if enc["input_ids"][:n] != list(ids):
            stats["ids_mismatch"].append(name + "/offsets")
        starts = np.array([a for a, _ in offs[:n]])
        ends = np.array([b for _, b in offs[:n]])

        conv_of[cursor:cursor + n] = ci
        conv_start[name] = cursor

        char_cursor = 0
        onset_tok = None
        oj = onsets.get(name, {})
        ai = -1
        for m in msgs:
            if m["role"] != "assistant":
                continue
            ai += 1
            cs = text.find(m["content"], char_cursor)
            if cs < 0:
                stats["span_fail"].append(f"{name}/{ai}")
                continue
            ce = cs + len(m["content"])
            char_cursor = ce
            tmask = (starts < ce) & (ends > cs)
            tidx = np.flatnonzero(tmask)
            if len(tidx) == 0:
                continue
            turn_of[cursor + tidx] = ai
            sc = scores.get(f"{name}/{ai}")
            if sc is not None:
                score_of[cursor + tidx] = sc
            pos_in_turn[cursor + tidx] = np.arange(len(tidx))
            if (onset_tok is None and isinstance(oj, dict)
                    and oj.get("turn_index") == ai
                    and oj.get("emotional_word")):
                w = oj["emotional_word"]
                pc = oj.get("preceding_context") or ""
                loc = m["content"].find(pc + w) if pc else -1
                wloc = (loc + len(pc)) if loc >= 0 else \
                    m["content"].find(w)
                if wloc < 0:
                    stats["onset_match_fail"].append(name)
                else:
                    gchar = cs + wloc
                    cand = np.flatnonzero((starts <= gchar)
                                          & (ends > gchar))
                    if len(cand):
                        onset_tok = cursor + int(cand[0])
        if onset_tok is not None:
            dist_onset[cursor:cursor + n] = onset_tok - np.arange(
                cursor, cursor + n)
        ids_all[cursor:cursor + n] = np.asarray(ids, dtype=np.int32)
        cursor += n
    assert cursor == N, (cursor, N)

    np.savez(LAB, conv_of=conv_of, turn_of=turn_of, score_of=score_of,
             dist_onset=dist_onset, pos_in_turn=pos_in_turn,
             ids=ids_all,
             conv_names=np.array(names),
             conv_starts=np.array([conv_start[n] for n in names]))
    return (names, conv_of, turn_of, score_of, dist_onset, pos_in_turn,
            ids_all, conv_start, stats)


def manifests(names, conv_of, turn_of, score_of, dist_onset, pos_in_turn,
              ids_all, conv_start, stats):
    N = len(conv_of)
    puzzles = sorted({n.rsplit("_", 1)[0] for n in names})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(puzzles))
    test_puz = {puzzles[i] for i in perm[:len(puzzles) // 5]}
    conv_test = np.array([names[c].rsplit("_", 1)[0] in test_puz
                          for c in range(len(names))])

    # depth = position within conversation (flat idx − conv start)
    starts_arr = np.zeros(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        starts_arr[i] = conv_start[n]
    depth = np.arange(N) - starts_arr[conv_of]

    is_asst = turn_of >= 0
    elig = is_asst & (depth >= P_MIN)
    has_onset = dist_onset != NO_ONSET
    pre = has_onset & (dist_onset > 0)
    far_neg = elig & ((~has_onset) | (pre & (dist_onset >= GUARD_HI)))

    def pit_bucket(v):
        return int(np.searchsorted(PIT_EDGES, v, side="right") - 1)

    out_m, out_s = {}, {}
    tasks = {}
    for tname, (lo, hi) in BANDS.items():
        pos = elig & pre & (dist_onset >= lo) & (dist_onset <= hi)
        tasks[tname] = {"classes": {1: np.flatnonzero(pos),
                                    0: np.flatnonzero(far_neg)},
                        "cell": lambda i: (int(ids_all[i]),
                                           int(turn_of[i])),
                        "n_classes": 2}
    sc_pre = elig & (score_of >= 0) & ((~has_onset) | (dist_onset > 0))
    v = score_of[sc_pre]
    if len(v):
        edges = np.quantile(v, [1 / 3, 2 / 3])
        terc = np.digitize(score_of.astype(float), edges)
        tasks["esc3"] = {
            "classes": {c: np.flatnonzero(sc_pre & (terc == c))
                        for c in range(3)},
            "cell": lambda i: (int(turn_of[i]),
                               pit_bucket(pos_in_turn[i])),
            "n_classes": 3}
        out_s["esc3_edges"] = [float(x) for x in edges]
    det_pos = elig & has_onset & (dist_onset < 0) \
        & (dist_onset >= -DET_HI)
    tasks["det"] = {"classes": {1: np.flatnonzero(det_pos),
                                0: np.flatnonzero(far_neg & pre)},
                    "cell": lambda i: (int(turn_of[i]), 0),
                    "n_classes": 2}

    import zlib
    for tname, spec in tasks.items():
        for split, want_test in [("train", False), ("test", True)]:
            pools = defaultdict(list)
            for c, idxs in spec["classes"].items():
                for i in idxs:
                    if conv_test[conv_of[i]] != want_test:
                        continue
                    t_id, hb = spec["cell"](int(i))
                    pools[c].append((int(i), 0, t_id, hb))
            rng = np.random.default_rng(MATCH_SEED + zlib.crc32(
                f"emo/{tname}/{split}".encode()) % 2 ** 16)
            out, joint = matched_sample(dict(pools), CAP[split], rng,
                                        spec["n_classes"])
            n_per = {int(c): len(vv) for c, vv in out.items()}
            out_s[f"{tname}/{split}"] = {
                "rows_per_class": n_per, "joint_matched": bool(joint),
                "ok": bool(min(n_per.values(), default=0) >= 300)}
            rows = np.array([r[0] for c in sorted(out)
                             for r in out[c]], dtype=np.int64)
            y = np.concatenate(
                [np.full(len(out[c]), c, dtype=np.int64)
                 for c in sorted(out)]) if len(rows) else \
                np.zeros(0, dtype=np.int64)
            out_m[f"{tname}_{split}_rows"] = rows
            out_m[f"{tname}_{split}_y"] = y
    np.savez(RES / "manifests.npz", **out_m)

    n_onset = int(((dist_onset == 0)).sum())
    out_s["summary"] = {
        "n_tokens": int(N), "n_convs": len(names),
        "n_convs_with_onset": n_onset,
        "onset_match_fail": stats["onset_match_fail"],
        "span_fail": stats["span_fail"][:20],
        "ids_mismatch": stats["ids_mismatch"][:20],
    }
    (RES / "label_stats.json").write_text(json.dumps(out_s, indent=1))
    for k, vv in out_s.items():
        if isinstance(vv, dict) and "rows_per_class" in vv:
            print(f"[labels] {k}: {vv['rows_per_class']} ok={vv['ok']}",
                  flush=True)
    print(f"[labels] onsets={n_onset} fails="
          f"{len(stats['onset_match_fail'])} -> {RES}", flush=True)


if __name__ == "__main__":
    manifests(*build())
