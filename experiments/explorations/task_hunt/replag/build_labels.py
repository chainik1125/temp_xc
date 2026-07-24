"""Repetition-lag Δ labels — exact, from tokens (CARD.md, frozen).

Built inline by runpod-e per `briefings/task-hunt-b.md` (runpod-b's
parallel builder had not landed when caches were ready). Everything is
deterministic given the committed fineweb sample + the frozen seeds.

Per model tokenizer: join each doc's sentences, tokenize, cut into
non-overlapping model-visible sequences of length 128 (BOS + 127
content for gemma/llama; 128 content for gpt2). For content position p,
Δn(p) = distance to the previous in-sequence occurrence of the current
n-gram (n ∈ {1, 2}), −1 if none. Emits:

  /workspace/replag_caches/<model>/tokens.npz   ids (N,128) i32, doc_idx
  /workspace/replag_caches/<model>/delta.npz    delta1, delta2 (N,128) i32
  experiments/explorations/task_hunt/labels/replag_<model>_manifests.npz
      per task/split: rows (n,2)=(seq,pos), y — anchor-identity ×
      position-bucket matched, doc-split 80/20, capped (CARD § controls)
  experiments/explorations/task_hunt/labels/replag_<model>_stats.json
      bucket counts, cov(B,T), shuffled-null Δ histogram, fallback
      flags, sanity-test results

Ships with 5 sanity tests (run every build; build fails on any):
  T1 brute-force vs fast Δ on random sequences (n = 1, 2)
  T2 definition check on real rows (match at p−Δ, none closer)
  T3 no cross-sequence leakage (Δ ≤ content history everywhere)
  T4 burstiness: real P(Δ≤4 | eligible) > shuffled-null
  T5 manifest integrity: exact class-histogram equality, eligibility,
     caps, determinism (rebuild → identical)

Run: .venv/bin/python -m experiments.explorations.task_hunt.replag.build_labels [model ...]
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = {
    "gpt2": {"hf": "openai-community/gpt2", "bos": False},
    "gemma2_2b": {"hf": "google/gemma-2-2b", "bos": True},
    "llama31_8b": {"hf": "NousResearch/Meta-Llama-3.1-8B", "bos": True},
}
SEQ_LEN = 128
H_MIN = 64                      # min content history at an anchor
NEG_MIN_DELTA = 65              # negative = Δ ≥ 65 or none
BUCKETS = {"B4": (2, 4), "B8": (5, 8), "B16": (9, 16), "B32": (17, 32)}
T_GRID = [2, 4, 8, 16, 32]
CAP = {"train": 4000, "test": 1500}
MIN_ROWS = 300
SPLIT_SEED = 7                  # doc split, probe_depth convention
MATCH_SEED = 1013
NULL_SEED = 271                 # shuffled-null label sanity
POS_EDGES = [64, 80, 96, 112, 128]

REPO = Path(__file__).resolve().parents[4]
SAMPLE = (REPO / "experiments/explorations/synthetic/expansion/data/"
          "fineweb_sample.json")
LABELS_DIR = REPO / "experiments/explorations/task_hunt/labels"
CACHE_ROOT = Path("/workspace/replag_caches")


# ---------------------------------------------------------------- tokenize

def tokenize_model(key: str):
    """Token grid: (N, 128) int32 + doc_idx (N,) + n_prefix."""
    from transformers import AutoTokenizer
    cfg = MODELS[key]
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    n_prefix = 1 if cfg["bos"] else 0
    content_len = SEQ_LEN - n_prefix
    docs = json.loads(SAMPLE.read_text())["docs"]
    seqs, doc_idx = [], []
    for di, doc in enumerate(docs):
        ids = tok(" ".join(doc["sentences"]),
                  add_special_tokens=False)["input_ids"]
        for s in range(0, len(ids) - content_len + 1, content_len):
            chunk = ids[s:s + content_len]
            if cfg["bos"]:
                chunk = [tok.bos_token_id] + chunk
            seqs.append(chunk)
            doc_idx.append(di)
    return (np.asarray(seqs, dtype=np.int32),
            np.asarray(doc_idx, dtype=np.int32), n_prefix)


# ------------------------------------------------------------------ deltas

def compute_delta(ids: np.ndarray, n_prefix: int, n: int) -> np.ndarray:
    """Δn per position (−1 = none / invalid). Dict scan per sequence."""
    N, L = ids.shape
    out = np.full((N, L), -1, dtype=np.int32)
    for i in range(N):
        row = ids[i]
        last: dict = {}
        for p in range(n_prefix + n - 1, L):
            k = int(row[p]) if n == 1 else (int(row[p - 1]), int(row[p]))
            q = last.get(k)
            if q is not None:
                out[i, p] = p - q
            last[k] = p
    return out


def compute_delta_brute(ids: np.ndarray, n_prefix: int, n: int) -> np.ndarray:
    N, L = ids.shape
    out = np.full((N, L), -1, dtype=np.int32)
    for i in range(N):
        row = ids[i]
        for p in range(n_prefix + n - 1, L):
            for q in range(p - 1, n_prefix + n - 2, -1):
                if all(row[q - j] == row[p - j] for j in range(n)):
                    out[i, p] = p - q
                    break
    return out


# ---------------------------------------------------------------- matching

def _cells(rows_per_class, joint: bool):
    """cell key = (token_id, H-bucket) [joint] or token_id [fallback]."""
    cells: dict = defaultdict(lambda: defaultdict(list))
    for c, rows in rows_per_class.items():
        for (s, p, t, hb) in rows:
            cells[(t, hb) if joint else t][c].append((s, p))
    return cells


def matched_sample(rows_per_class: dict, cap: int, rng,
                   n_classes: int) -> tuple[dict, bool]:
    """Exact class-histogram matching over cells; seeded cell order;
    same per-cell take for every class (preserves equality under cap).
    Falls back to token-only cells if joint matching is too thin."""
    classes = sorted(rows_per_class)
    if len(classes) != n_classes:          # a class pool is empty
        return {c: [] for c in range(n_classes)}, False
    for joint in (True, False):
        cells = _cells(rows_per_class, joint)
        keys = sorted(cells.keys())
        order = rng.permutation(len(keys))
        out = {c: [] for c in classes}
        total = 0
        for ki in order:
            ck = keys[ki]
            m = min(len(cells[ck].get(c, [])) for c in classes)
            m = min(m, cap - total)
            if m <= 0:
                if total >= cap:
                    break
                continue
            for c in classes:
                lst = cells[ck][c]
                take = rng.choice(len(lst), size=m, replace=False)
                out[c].extend(lst[i] for i in take)
            total += m
        if total >= MIN_ROWS:
            return out, joint
    return out, False        # thin either way; caller records/raises


def build_manifests(ids, doc_idx, delta1, n_prefix, seed=MATCH_SEED):
    """All task manifests for one model. Returns dict + stats."""
    N, L = ids.shape
    H = np.arange(L) - n_prefix                       # content history
    elig_p = np.where(H >= H_MIN)[0]

    docs = np.unique(doc_idx)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(docs))
    test_docs = set(docs[perm[:len(docs) // 5]].tolist())
    is_test_seq = np.array([doc_idx[i] in test_docs for i in range(N)])

    def hbucket(p):
        return int(np.searchsorted(POS_EDGES, H[p], side="right") - 1)

    # candidate rows: class pools per task, per split
    def gather(split):
        want_test = split == "test"
        pools: dict = {t: defaultdict(list) for t in
                       [f"det{b[1:]}" for b in BUCKETS] + ["lag4"]}
        for i in range(N):
            if is_test_seq[i] != want_test:
                continue
            for p in elig_p:
                d = int(delta1[i, p])
                t_id, hb = int(ids[i, p]), hbucket(p)
                row = (i, int(p), t_id, hb)
                if d == -1 or d >= NEG_MIN_DELTA:
                    for bname in BUCKETS:
                        pools[f"det{bname[1:]}"][0].append(row)
                else:
                    for bi, (bname, (lo, hi)) in enumerate(BUCKETS.items()):
                        if lo <= d <= hi:
                            pools[f"det{bname[1:]}"][1].append(row)
                            pools["lag4"][bi].append(row)
        return pools

    manifests, stats = {}, {}
    for split in ["train", "test"]:
        pools = gather(split)
        for task, per_class in pools.items():
            rng = np.random.default_rng(
                seed + zlib.crc32(f"{task}/{split}".encode()) % (2 ** 16))
            out, joint = matched_sample(dict(per_class), CAP[split], rng,
                                        4 if task == "lag4" else 2)
            n_per = {int(c): len(v) for c, v in out.items()}
            n0 = min(n_per.values()) if n_per else 0
            stats[f"{task}/{split}"] = {
                "rows_per_class": n_per,
                "joint_matched": bool(joint),
                "ok": bool(n0 >= MIN_ROWS)}
            rows = np.array([r for c in sorted(out) for r in out[c]],
                            dtype=np.int32)
            y = np.concatenate([np.full(len(out[c]), c, dtype=np.int8)
                                for c in sorted(out)]) if len(rows) else \
                np.zeros(0, dtype=np.int8)
            manifests[f"{task}_{split}_rows"] = rows
            manifests[f"{task}_{split}_y"] = y
    return manifests, stats


# ------------------------------------------------------------------ sanity

def sanity(ids, doc_idx, delta1, delta2, n_prefix, manifests, stats, key):
    res = {}
    rng = np.random.default_rng(0)

    # T1 brute vs fast on random sequences
    fake = rng.integers(0, 50, size=(60, 40)).astype(np.int32)
    for n, name in [(1, "d1"), (2, "d2")]:
        a = compute_delta(fake, 1, n)
        b = compute_delta_brute(fake, 1, n)
        assert np.array_equal(a, b), f"T1 {name}"
    res["T1_brute_vs_fast"] = "pass"

    # T2 definition on real rows
    N, L = ids.shape
    cand = np.argwhere(delta1 >= 2)
    sel = cand[rng.choice(len(cand), size=min(500, len(cand)),
                          replace=False)]
    for i, p in sel:
        d = delta1[i, p]
        assert ids[i, p - d] == ids[i, p], "T2 match"
        assert not np.any(ids[i, p - d + 1:p] == ids[i, p]), "T2 closer"
    res["T2_definition"] = "pass"

    # T3 no cross-sequence leakage
    H = np.arange(L) - n_prefix
    labeled = delta1 >= 1
    assert np.all(delta1[labeled] <= np.broadcast_to(H, (N, L))[labeled]), "T3"
    res["T3_no_leakage"] = "pass"

    # T4 burstiness vs within-sequence shuffled null
    nrng = np.random.default_rng(NULL_SEED)
    shuf = ids.copy()
    for i in range(N):
        c = shuf[i, n_prefix:].copy()
        nrng.shuffle(c)
        shuf[i, n_prefix:] = c
    d_null = compute_delta(shuf, n_prefix, 1)
    epos = np.zeros((N, L), dtype=bool)
    epos[:, H_MIN + n_prefix:] = True
    # Two-sided structure test: the real Δ distribution must diverge
    # from the exchangeable (bag-preserving) null. Direction is a
    # recorded FINDING, not an assumption — first build showed real
    # text has FEWER Δ≤4 repeats than its shuffled bag (grammar avoids
    # near-repetition; uniform placement clumps), falsifying the card's
    # parenthetical prior (amended pre-screen, see LOG).
    real_p4 = float(np.mean(
        (delta1 >= 2)[epos] & (delta1 <= 4)[epos]))
    null_p4 = float(np.mean(
        (d_null >= 2)[epos] & (d_null <= 4)[epos]))
    hist_real = [int(((delta1 == d) & epos).sum()) for d in range(1, 65)]
    hist_null = [int(((d_null == d) & epos).sum()) for d in range(1, 65)]
    n_e = int(epos.sum())
    pr = np.array(hist_real + [n_e - sum(hist_real)]) / n_e
    pn = np.array(hist_null + [n_e - sum(hist_null)]) / n_e
    tv = float(0.5 * np.abs(pr - pn).sum())
    assert tv > 0.02, f"T4 real≈null (TV={tv:.4f})"
    res["T4_null_divergence"] = {"tv_distance": tv,
                                 "real_p_delta_le4": real_p4,
                                 "null_p_delta_le4": null_p4,
                                 "direction": ("real<null at Δ≤4"
                                               if real_p4 < null_p4
                                               else "real>null at Δ≤4"),
                                 "hist_real_1_64": hist_real,
                                 "hist_null_1_64": hist_null}

    # T5 manifest integrity + determinism
    for task in [f"det{b[1:]}" for b in BUCKETS] + ["lag4"]:
        for split in ["train", "test"]:
            rows = manifests[f"{task}_{split}_rows"]
            y = manifests[f"{task}_{split}_y"]
            if not stats[f"{task}/{split}"]["ok"]:
                continue
            hists = {}
            for c in np.unique(y):
                m = y == c
                keys = [(int(ids[s, p]),
                         int(np.searchsorted(POS_EDGES, p - n_prefix,
                                             side="right") - 1))
                        for s, p in rows[m]]
                hists[int(c)] = sorted(keys)
                assert np.all(rows[m][:, 1] - n_prefix >= H_MIN), "T5 elig"
                assert m.sum() <= CAP[split], "T5 cap"
            ref = hists[min(hists)]
            assert all(h == ref for h in hists.values()), f"T5 hist {task}"
    m2, _ = build_manifests(ids, doc_idx, delta1, n_prefix)
    for k in manifests:
        assert np.array_equal(manifests[k], m2[k]), f"T5 determinism {k}"
    res["T5_manifests"] = "pass"
    return res


# ---------------------------------------------------------------- coverage

def coverage(manifests, delta1):
    """cov(B, T) = P(Δ ≤ T−1 | Δ ∈ B) on matched det positives (train)."""
    out = {}
    for bname, (lo, hi) in BUCKETS.items():
        task = f"det{bname[1:]}"
        rows = manifests[f"{task}_train_rows"]
        y = manifests[f"{task}_train_y"]
        d = np.array([delta1[s, p] for s, p in rows[y == 1]])
        out[bname] = {f"T{t}": (float(np.mean(d <= t - 1)) if len(d) else
                                None) for t in T_GRID}
    return out


# -------------------------------------------------------------------- main

def build(key: str):
    t0 = time.time()
    out_dir = CACHE_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    ids, doc_idx, n_prefix = tokenize_model(key)
    print(f"[{key}] {ids.shape[0]} seqs × {SEQ_LEN} "
          f"({time.time() - t0:.0f}s tokenize)", flush=True)
    delta1 = compute_delta(ids, n_prefix, 1)
    delta2 = compute_delta(ids, n_prefix, 2)
    np.savez(out_dir / "tokens.npz", ids=ids, doc_idx=doc_idx,
             n_prefix=np.int32(n_prefix))
    np.savez(out_dir / "delta.npz", delta1=delta1, delta2=delta2)

    manifests, stats = build_manifests(ids, doc_idx, delta1, n_prefix)
    tests = sanity(ids, doc_idx, delta1, delta2, n_prefix,
                   manifests, stats, key)
    cov = coverage(manifests, delta1)

    np.savez(LABELS_DIR / f"replag_{key}_manifests.npz", **manifests)
    (LABELS_DIR / f"replag_{key}_stats.json").write_text(json.dumps({
        "model": MODELS[key]["hf"], "n_seqs": int(ids.shape[0]),
        "seq_len": SEQ_LEN, "n_prefix": n_prefix, "h_min": H_MIN,
        "buckets": BUCKETS, "neg_min_delta": NEG_MIN_DELTA,
        "caps": CAP, "seeds": {"split": SPLIT_SEED, "match": MATCH_SEED,
                               "null": NULL_SEED},
        "tasks": stats, "coverage": cov, "sanity": tests,
        "wall_seconds": round(time.time() - t0, 1)}, indent=1))
    for k, v in stats.items():
        print(f"  [{key}] {k}: {v['rows_per_class']} "
              f"joint={v['joint_matched']} ok={v['ok']}", flush=True)
    print(f"[{key}] DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(MODELS)):
        build(key)
