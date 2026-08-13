"""Survivor-latent decomposition for the w6mix DSM dictionary. In-container.

Question (the interesting-vs-trivial fork behind detection_controls.json):
are w6mix_dsm's ~849 surviving latents individually meaningful features
("higher-level, hence fewer needed"), or a strongly-interacting distributed
code ("densification exploiting interactions, interpretability given up")?

Quantitative frame both readings must confront: dsm and recon both run
saturated at L0 ~= 96 actives/window, so survivor firing rate ~= 96/849 ~=
11.3% vs recon-live ~= 96/15274 ~= 0.63%; the measured ~18x per-latent
informativeness IS the rate ratio. The open question is per-FIRING-EVENT
semantics, probed three ways through the identical capture / sentence
example-set / probe path of run_detection (dictionary side only changes):

  1. interaction structure
     (a) single-latent probes: detect_core.probe_cv on ONE feature column
         (S=1; with one column the top-S Welch selection is the identity, so
         this is the same protocol restricted to that latent) for every dsm
         survivor and for recon's top-849-by-firing-rate latents.
     (b) subset-capacity curve: probe_cv (S in {8,32}) on random subsets of
         the survivor pool at sizes {1,4,16,64,256,849}, 3 draws each; same
         sizes drawn from recon's live pool for comparison.
     (c) pairwise activation correlation: Pearson |corr| among survivor
         columns vs among recon's top-849 columns, on a row-strided sample
         of all stride-1 trace windows (rates differ ~18x between groups, so
         the raw corr distributions carry a rate-driven component; both are
         reported with their group mean rates).
  2. rate-normalised information: probe performance vs total firing mass.
     recon live latents sorted by rate, sliced at cumulative-mass fractions
     {.25,.5,.75,.9,.95,1.0} of recon's total mass, plus the slice matched
     to dsm's TOTAL survivor mass (~96/window) and the count-matched N=849
     slice; dsm's own mass curve for symmetry. NOTE the arithmetic identity:
     both dicts saturate at 96, so "recon matched to dsm's total mass" is
     expected to be ~the whole recon live pool -- if so, that IS the
     finding, and the informative content is the shape of recon's
     info-vs-mass curve against dsm's single point.
  3. judged autointerp on window units: 50 dsm survivors (rate-stratified
     quintiles), 50 recon top-usage, 50 recon random-live; per latent the
     top-12 max-activating stride-1 windows (deduped: >=12-token separation
     within a trace, <=3 windows/trace, trace-boundary windows excluded),
     rendered as text with the 6-token span marked and ~2 tokens context
     each side; judge = the run_evals.py::autointerp_judge protocol adapted
     to window units (explanation from the top 8, precision test on the
     held-out 4 vs 4 decoys from other latents' dumps, balanced-accuracy
     score), claude-haiku via the em-sprint-judges key.

PRE-REGISTERED READINGS (written before any results existed):
  interesting branch ("fewer, higher-level features"):
    - single-latent signal substantial: top survivors well above the
      label-shuffle floor (~0.144 sentence-S8 PR-AUC) alone;
    - capacity curve rises smoothly from small subset sizes;
    - autointerp(dsm survivors) >= autointerp(recon top-usage);
    - pairwise |corr| comparable between survivor and recon groups.
  trivial branch ("interaction-borne distributed code"):
    - individual survivors ~= chance while the joint probe is high;
      capacity curve near-chance until large sizes, then a cliff;
    - autointerp clearly worse for dsm survivors;
    - pairwise |corr| clearly higher among survivors.
  measurement 2: matched-mass recon ~= dsm  => per-event information equal
    (trivial reading of the 18x); dsm >> matched-mass recon => DSM firing
    events are individually richer.
  Mixed outcomes are reported per-measurement; no forced verdict.

Writes /vol/backtracking_eval/survivor_interp{suffix}.json (committed after
every block -- the volume artifact is the launch evidence) and per-group
max-activating-window dumps under /vol/backtracking_eval/survivor_interp/.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
import urllib.request

import numpy as np

from experiments.backtracking_detection_dsm import detect_core as dc

H = 16384
K = 96
T = len(dc.WIN_OFF)
W6MIX_FILES = [("dsm", "txc_dsm/dsm_s0.pt"), ("recon", "txc_recon/recon_s0.pt")]
SHUFFLE_FLOOR_S8 = 0.144      # detection_controls.json label-shuffle mean
JUDGE_MODEL = "claude-haiku-4-5-20251001"


# --------------------------------------------------------------------------
# sweep helpers (same time-major stride-1 windowing as detect_core)
# --------------------------------------------------------------------------

def sweep_collect(model, cache_hook: np.ndarray, cols: np.ndarray,
                  device="cuda", step=4096):
    """One full pass over every stride-1 time-major T-window of the trace
    cache, collecting the given latent columns' post-TopK z values (fp32) and
    the sweep-mean L0. Windowing identical to detect_core.w6_fire_counts."""
    import torch

    d_flat = cache_hook.shape[1] * T
    n_win = cache_hook.shape[0] - T + 1
    cols_t = torch.from_numpy(np.asarray(cols, dtype=np.int64)).to(device)
    out = np.empty((n_win, len(cols)), dtype=np.float32)
    l0_sum = 0.0
    with torch.no_grad():
        for s in range(0, n_win, step):
            m = min(step, n_win - s)
            block = np.asarray(cache_hook[s:s + m + T - 1])
            w = np.lib.stride_tricks.sliding_window_view(block, T, axis=0)
            x = torch.from_numpy(np.ascontiguousarray(
                w.transpose(0, 2, 1)).reshape(m, d_flat)).to(
                device, torch.float32)
            z = model.encode(x)
            l0_sum += float((z > 0).sum())
            out[s:s + m] = z[:, cols_t].cpu().numpy()
    return out, l0_sum / max(n_win, 1)


def valid_window_mask(offsets: np.ndarray, n_win: int) -> np.ndarray:
    """True where the stride-1 window [s, s+T-1] lies inside one trace."""
    s = np.arange(n_win)
    k = np.searchsorted(offsets, s, side="right") - 1
    return (s + T - 1) < offsets[k + 1]


def top_windows(vals: np.ndarray, offsets: np.ndarray, valid: np.ndarray,
                n_take=12, min_gap=12, per_trace_cap=3, scan_cap=40000):
    """Top-activating windows with de-duplication: only within-trace windows,
    >= min_gap tokens apart inside a trace, <= per_trace_cap per trace."""
    order = np.argsort(-vals)[:scan_cap]
    sel, per_trace, in_trace = [], {}, {}
    for s in order:
        if vals[s] <= 0 or not valid[s]:
            continue
        k = int(np.searchsorted(offsets, s, side="right") - 1)
        if per_trace.get(k, 0) >= per_trace_cap:
            continue
        if any(abs(int(s) - p) < min_gap for p in in_trace.get(k, ())):
            continue
        sel.append((int(s), k, float(vals[s])))
        per_trace[k] = per_trace.get(k, 0) + 1
        in_trace.setdefault(k, []).append(int(s))
        if len(sel) >= n_take:
            break
    return sel


# --------------------------------------------------------------------------
# pairwise correlation
# --------------------------------------------------------------------------

def paircorr_stats(M: np.ndarray, min_nnz=20) -> dict:
    """Off-diagonal |Pearson corr| statistics over the columns of M.
    Columns with < min_nnz nonzeros in the sample are dropped: the smoke run
    showed that 2-3-event columns yield degenerate |corr|=1.0 pairs that
    dominate the tail statistics with pure noise (fix pre-registered before
    the full run's results existed)."""
    nnz = (M > 0).sum(0)
    keep = nnz >= min_nnz
    X = M[:, keep].astype(np.float64)
    X -= X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X /= sd
    C = (X.T @ X) / X.shape[0]
    iu = np.triu_indices(C.shape[1], k=1)
    a = np.abs(C[iu])
    a_sorted = np.sort(a)
    n_top = max(1, int(0.01 * len(a)))
    return {
        "n_latents_kept": int(keep.sum()),
        "n_latents_dropped_lt_min_nnz": int((~keep).sum()),
        "n_rows": int(M.shape[0]),
        "n_pairs": int(len(a)),
        "mean_abs_corr": float(a.mean()),
        "median_abs_corr": float(np.median(a)),
        "p99_abs_corr": float(np.quantile(a, 0.99)),
        "top1pct_mean_abs_corr": float(a_sorted[-n_top:].mean()),
        "max_abs_corr": float(a_sorted[-1]),
    }


# --------------------------------------------------------------------------
# autointerp rendering + judge
# --------------------------------------------------------------------------

class TraceRenderer:
    """Re-tokenizes traces (same tokenizer + add_special_tokens=False as
    capture_traces) lazily, and renders a window [p, p+T-1] of trace k as
    text with the window span marked and ctx tokens of context each side."""

    def __init__(self, model_id: str, traces_path: str, trace_meta):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        traces = json.loads(pathlib.Path(traces_path).read_text())
        self.by_qid = {t["question_id"]: t for t in traces}
        self.meta = trace_meta
        self._cache: dict = {}

    def _spans(self, k: int):
        if k not in self._cache:
            full = self.by_qid[self.meta[k]["qid"]]["full_response"]
            enc = self.tok(full, return_offsets_mapping=True,
                           add_special_tokens=False)
            spans = enc["offset_mapping"]
            if len(spans) != self.meta[k]["n_tokens"]:
                print(f"[render WARN] trace {k}: retokenized "
                      f"{len(spans)} != captured "
                      f"{self.meta[k]['n_tokens']} tokens", flush=True)
            self._cache[k] = (full, spans)
        return self._cache[k]

    def render(self, k: int, p: int, ctx: int = 2) -> str:
        full, spans = self._spans(k)
        n = len(spans)
        lo, hi = max(0, p - ctx), min(n - 1, p + T - 1 + ctx)
        c0 = spans[lo][0]
        cs, ce = spans[p][0], spans[min(p + T - 1, n - 1)][1]
        c1 = spans[hi][1]
        return (full[c0:cs] + "«" + full[cs:ce] + "»" + full[ce:c1]
                ).replace("\n", " ")


def _judge_call(prompt: str, key: str, model: str = JUDGE_MODEL,
                retries: int = 3) -> str:
    body = json.dumps({"model": model, "max_tokens": 300,
                       "messages": [{"role": "user", "content": prompt}],
                       }).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["content"][0]["text"]
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"judge call failed after {retries} tries: {last}")


def judge_latent(lat: int, examples: list[dict], decoys: list[str],
                 key: str, rng: np.random.Generator) -> dict:
    """run_evals.py::autointerp_judge protocol on window units: explanation
    from the top 8 examples, precision test on held-out ranks 9-12 vs 4
    decoys, balanced accuracy 0.5*(tp/4 + (4-fp)/4)."""
    top, held = examples[:8], examples[8:12]
    expl = _judge_call(
        "Each snippet below is text from a chain-of-thought reasoning "
        "trace. In each one, the 6-token span marked between « and "
        "» strongly activates the same feature of a sparse dictionary "
        "trained on sliding 6-token windows of a language model's "
        "activations. Write a ONE-LINE explanation of what this feature "
        "responds to.\n\n"
        + "\n---\n".join(e["ctx"][-300:] for e in top), key)
    test = [e["ctx"] for e in held] + decoys
    order = rng.permutation(len(test))
    listing = "\n".join(f"[{i}] {test[j][-300:]}"
                        for i, j in enumerate(order))
    verdict = _judge_call(
        f"Feature explanation: {expl}\nWhich of these 8 snippets contain a "
        "«»-marked span that activates this feature? Reply with "
        f"ONLY the bracket numbers, comma-separated.\n\n{listing}", key)
    truth = {i for i, j in enumerate(order) if j < len(held)}
    said = {int(m) for m in re.findall(r"\d+", verdict) if int(m) < 8}
    tp = len(truth & said)
    fp = min(len(said - truth), 4)
    return {"latent": int(lat), "score": 0.5 * (tp / 4 + (4 - fp) / 4),
            "explanation": expl.strip(), "verdict_raw": verdict.strip(),
            "tp": tp, "fp": fp}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(device: str = "cuda", vol: str = "/vol",
        traces_path: str = "/work/data/traces.json",
        labels_path: str = "/work/data/sentence_labels.json",
        limit_traces: int | None = None, tag_suffix: str = "",
        commit_cb=None, w6_dir: str = "/vol/txc_w6_mix",
        n_auto: int = 50, n_top_windows: int = 12, min_fires: int = 30,
        subset_sizes: tuple[int, ...] = (1, 4, 16, 64, 256, 849),
        n_draws: int = 3, corr_row_stride: int = 5,
        max_single: int | None = None, seed: int = 20260813,
        do_judge: bool = True) -> dict:
    import os

    import torch

    t0 = time.time()
    out: dict = {"protocol": {
        "window_offsets": dc.WIN_OFF,
        "probe": "l1-logistic C=1 liblinear, 5-fold GroupKFold by trace, "
                 "top-S by train-fold Welch t-stat (detect_core.probe_cv, "
                 "unchanged); single-latent arms use S=1 (identity "
                 "selection on one column)",
        "score": "window latent |z| (w6_flat time-major flatten, unchanged)",
        "example_set": "sentence only (the reference S8 numbers' set); "
                       "far omitted to keep the per-latent loops tractable",
        "labels": "results/ward_backtracking/sentence_labels.json",
        "w6_dir": w6_dir,
        "seed": seed, "min_fires_for_autointerp": min_fires,
        "n_auto_per_group": n_auto, "n_top_windows": n_top_windows,
        "corr_row_stride": corr_row_stride,
        "judge_model": JUDGE_MODEL,
        "preregistered_readings": {
            "interesting": "top survivors alone well above the "
                           f"label-shuffle floor (~{SHUFFLE_FLOOR_S8} "
                           "sentence-S8); smooth capacity curve; "
                           "autointerp(dsm) >= autointerp(recon top-usage); "
                           "pairwise |corr| comparable between groups",
            "trivial": "single survivors ~= chance with joint high (cliff "
                       "capacity curve); autointerp clearly worse for dsm; "
                       "pairwise |corr| clearly higher among survivors",
            "m2": "matched-mass recon ~= dsm => per-event info equal "
                  "(trivial reading of the 18x); dsm >> matched-mass recon "
                  "=> DSM events individually richer. Identity note: both "
                  "dicts saturate at L0~=96 so the dsm-mass-matched recon "
                  "slice is expected to be ~the whole recon live pool.",
            "mixed": "report per-measurement, no forced verdict",
        },
    }}

    def flush():
        d = pathlib.Path(vol) / "backtracking_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"survivor_interp{tag_suffix}.json").write_text(
            json.dumps(out, indent=1, default=str))
        if commit_cb is not None:
            commit_cb()

    rng = np.random.default_rng(seed)

    # ---- capture (identical to run_controls) -----------------------------
    cache, offsets, trace_meta, cap_meta = dc.capture_traces(
        traces_path, labels_path, device=device, limit_traces=limit_traces)
    del cache["ln1"]                       # w6 dicts read resid_L10 only
    out["capture"] = cap_meta
    out["capture"]["n_traces"] = len(trace_meta)

    ex = dc.collect_examples(trace_meta)
    y = np.array([e[2] for e in ex["sentence"]])
    groups = np.array([e[0] for e in ex["sentence"]])
    out["example_sets"] = {"sentence": {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "positive_base_rate": float(y.mean()),
        "n_groups": int(len(set(groups.tolist())))}}
    print(f"[examples/sentence] n={len(y)} pos={int(y.sum())} "
          f"base_rate={y.mean():.4f}", flush=True)
    W = dc.gather_windows(cache["resid"], offsets, ex["sentence"])
    print(f"[windows/resid] sentence={tuple(W.shape)}", flush=True)
    flush()

    # ---- dictionaries + flatten-order gate -------------------------------
    models, metas = {}, {}
    for kind, fname in W6MIX_FILES:
        path = str(pathlib.Path(w6_dir) / fname)
        models[kind], metas[kind] = dc.load_w6_dict(path, kind, device)
    chk = dc.w6_nmse_both_orders(models["recon"], W, device)
    out["w6_ordering_check"] = chk
    print(f"[w6 ordering] NMSE time_major={chk['nmse_time_major']:.4f} "
          f"dim_major={chk['nmse_dim_major']:.4f}", flush=True)
    if not chk["nmse_time_major"] < 0.9 * chk["nmse_dim_major"]:
        flush()
        raise RuntimeError("w6 flatten-order gate failed; not scoring")

    # ---- fire counts / rates / pools -------------------------------------
    n_win = cache["resid"].shape[0] - T + 1
    counts, rates, live = {}, {}, {}
    out["live_pools"] = {}
    for kind in ("dsm", "recon"):
        c = dc.w6_fire_counts(models[kind], cache["resid"], device)
        counts[kind] = c
        rates[kind] = c / max(n_win, 1)
        live[kind] = np.where(c > 0)[0]
        out["live_pools"][kind] = {
            "n_live": int(len(live[kind])),
            "dead_fraction": float((c == 0).mean()),
            "n_windows": int(n_win), "unit": "stride1_window",
            "total_mass_sum_rates": float(rates[kind].sum()),
            "live_rate_mean": float(rates[kind][live[kind]].mean()),
            "live_rate_median": float(np.median(rates[kind][live[kind]])),
        }
        print(f"[live/{kind}] {len(live[kind])}/{H} live, "
              f"mass={rates[kind].sum():.2f}, mean live rate "
              f"{rates[kind][live[kind]].mean():.4f}", flush=True)
    survivors = live["dsm"]
    n_surv = len(survivors)
    recon_by_rate = live["recon"][np.argsort(-counts["recon"][live["recon"]])]
    dsm_total_mass = float(rates["dsm"][survivors].sum())
    flush()

    # ---- sentence feature matrices + headline re-check -------------------
    F = {kind: dc.window_features(models[kind], "w6_flat", W, device)
         for kind in ("dsm", "recon")}
    out["arms_recheck"] = {}
    for kind in ("dsm", "recon"):
        r = dc.probe_cv(F[kind], y, groups)
        r["sentence_l0_mean"] = float((F[kind] > 0).sum(1).mean())
        out["arms_recheck"][f"w6mix_{kind}"] = r
        print(f"[recheck w6mix_{kind}] S8={r['prauc_S8']:.4f} "
              f"S32={r['prauc_S32']:.4f} l0={r['sentence_l0_mean']:.2f}",
              flush=True)
    flush()

    # ---- measurement 2: info vs total firing mass ------------------------
    m2: dict = {"dsm_total_mass": dsm_total_mass,
                "recon_total_mass": float(rates["recon"].sum()),
                "recon_slices": [], "dsm_slices": []}
    fracs = (0.25, 0.5, 0.75, 0.9, 0.95, 1.0)

    def mass_slice(kind, ordered_idx, target_mass, label):
        cum = np.cumsum(rates[kind][ordered_idx])
        n = int(np.searchsorted(cum, target_mass) + 1)
        n = min(n, len(ordered_idx))
        cols = np.sort(ordered_idx[:n])
        r = dc.probe_cv(np.ascontiguousarray(F[kind][:, cols]), y, groups)
        rec = {"label": label, "n_latents": int(n),
               "mass_sum": float(cum[n - 1]),
               "mass_frac_of_dict": float(cum[n - 1] / max(cum[-1], 1e-9)),
               "prauc_S8": r["prauc_S8"], "prauc_S32": r["prauc_S32"],
               "rocauc_S8": r["rocauc_S8"]}
        print(f"[m2 {kind}/{label}] n={n} mass={rec['mass_sum']:.2f} "
              f"S8={r['prauc_S8']:.4f} S32={r['prauc_S32']:.4f}", flush=True)
        return rec

    dsm_by_rate = survivors[np.argsort(-counts["dsm"][survivors])]
    for f_ in fracs:
        m2["recon_slices"].append(mass_slice(
            "recon", recon_by_rate, f_ * float(rates["recon"].sum()),
            f"massfrac_{f_}"))
        out["m2_mass"] = m2
        flush()
    # the two pre-registered target slices
    m2["recon_slices"].append(mass_slice(
        "recon", recon_by_rate, dsm_total_mass, "matched_to_dsm_mass"))
    n_cm = min(n_surv, len(recon_by_rate))
    cols_cm = np.sort(recon_by_rate[:n_cm])
    r_cm = dc.probe_cv(np.ascontiguousarray(F["recon"][:, cols_cm]), y, groups)
    m2["recon_slices"].append({
        "label": f"count_matched_top{n_cm}_by_rate", "n_latents": int(n_cm),
        "mass_sum": float(rates["recon"][cols_cm].sum()),
        "mass_frac_of_dict": float(rates["recon"][cols_cm].sum()
                                   / max(rates["recon"].sum(), 1e-9)),
        "prauc_S8": r_cm["prauc_S8"], "prauc_S32": r_cm["prauc_S32"],
        "rocauc_S8": r_cm["rocauc_S8"]})
    print(f"[m2 recon/count_matched] n={n_cm} "
          f"mass={rates['recon'][cols_cm].sum():.2f} "
          f"S8={r_cm['prauc_S8']:.4f}", flush=True)
    for f_ in fracs:
        m2["dsm_slices"].append(mass_slice(
            "dsm", dsm_by_rate, f_ * dsm_total_mass, f"massfrac_{f_}"))
    out["m2_mass"] = m2
    flush()

    # ---- measurement 1a: single-latent probes ----------------------------
    def single_latent_block(kind, idx_pool, tag):
        idx_pool = np.asarray(idx_pool)
        if max_single is not None:
            idx_pool = idx_pool[:max_single]
        per = []
        tA = time.time()
        for j, lat in enumerate(idx_pool):
            r = dc.probe_cv(np.ascontiguousarray(F[kind][:, [lat]]),
                            y, groups, S_values=(1,))
            per.append({"latent": int(lat),
                        "rate": float(rates[kind][lat]),
                        "prauc_S1": r["prauc_S1"],
                        "rocauc_S1": r["rocauc_S1"]})
            if (j + 1) % 200 == 0:
                print(f"  [{tag}] {j + 1}/{len(idx_pool)} "
                      f"({time.time() - tA:.0f}s)", flush=True)
        pr = np.array([p["prauc_S1"] for p in per])
        srt = np.sort(pr)[::-1]
        summ = {
            "n": int(len(pr)), "max": float(srt[0]),
            "top10_mean": float(srt[:10].mean()),
            "median": float(np.median(pr)), "mean": float(pr.mean()),
            "frac_above_shuffle_floor": float((pr > SHUFFLE_FLOOR_S8).mean()),
            "n_above_shuffle_floor": int((pr > SHUFFLE_FLOOR_S8).sum()),
            "frac_above_base_rate": float(
                (pr > out["example_sets"]["sentence"]["positive_base_rate"])
                .mean()),
            "top20": [round(v, 4) for v in srt[:20]],
        }
        print(f"[m1a {tag}] max={summ['max']:.4f} "
              f"top10={summ['top10_mean']:.4f} med={summ['median']:.4f} "
              f">floor: {summ['n_above_shuffle_floor']}/{summ['n']}",
              flush=True)
        return {"summary": summ, "per_latent": per}

    out["m1a_single_latent"] = {}
    out["m1a_single_latent"]["dsm_survivors"] = single_latent_block(
        "dsm", survivors, "dsm_survivors")
    flush()
    out["m1a_single_latent"][f"recon_top{n_surv}_by_rate"] = \
        single_latent_block("recon", recon_by_rate[:n_surv], "recon_top")
    flush()

    # ---- measurement 1b: subset-capacity curves --------------------------
    sizes = sorted({min(s, n_surv) for s in subset_sizes} | {n_surv})
    out["m1b_capacity"] = {"sizes": sizes, "n_draws": n_draws, "curves": {}}
    for tag, kind, pool in (("dsm_survivors", "dsm", survivors),
                            ("recon_live", "recon", live["recon"])):
        curve = []
        for size in sizes:
            if size > len(pool):
                continue
            draws = []
            nd = 1 if size == len(pool) else n_draws
            for i in range(nd):
                sub_rng = np.random.default_rng(seed + 1000 * size + i)
                cols = np.sort(sub_rng.choice(pool, size=size, replace=False))
                r = dc.probe_cv(np.ascontiguousarray(F[kind][:, cols]),
                                y, groups)
                draws.append({"prauc_S8": r["prauc_S8"],
                              "prauc_S32": r["prauc_S32"]})
            curve.append({
                "size": int(size), "n_draws": nd,
                "prauc_S8_mean": float(np.mean([d["prauc_S8"]
                                                for d in draws])),
                "prauc_S8_draws": [round(d["prauc_S8"], 4) for d in draws],
                "prauc_S32_mean": float(np.mean([d["prauc_S32"]
                                                 for d in draws]))})
            print(f"[m1b {tag}] size={size} "
                  f"S8={curve[-1]['prauc_S8_mean']:.4f} "
                  f"({curve[-1]['prauc_S8_draws']})", flush=True)
            out["m1b_capacity"]["curves"][tag] = curve
            flush()
    del F

    # ---- full-sweep column collection (corr + autointerp) ----------------
    # autointerp latent selection first, so the recon sweep collects exactly
    # the columns needed.
    def eligible(kind, pool):
        return pool[counts[kind][pool] >= min_fires]

    el_surv = eligible("dsm", survivors)
    el_rec = eligible("recon", live["recon"])
    el_rec_by_rate = el_rec[np.argsort(-counts["recon"][el_rec])]

    # dsm: rate-stratified quintiles over eligible survivors
    srt_surv = el_surv[np.argsort(rates["dsm"][el_surv])]
    strata = np.array_split(srt_surv, 5)
    dsm_sample = []
    per_str = max(1, n_auto // 5)
    for st in strata:
        if not len(st):
            continue
        take = min(per_str, len(st))
        dsm_sample.extend(rng.choice(st, size=take, replace=False).tolist())
    dsm_sample = np.array(sorted(set(dsm_sample)))
    rec_top_sample = el_rec_by_rate[:n_auto]
    rest = np.setdiff1d(el_rec, rec_top_sample)
    rec_rand_sample = (np.sort(rng.choice(
        rest, size=min(n_auto, len(rest)), replace=False))
        if len(rest) else np.array([], dtype=np.int64))
    out["m3_selection"] = {
        "min_fires": min_fires,
        "n_eligible_dsm": int(len(el_surv)),
        "n_eligible_recon": int(len(el_rec)),
        "dsm_survivors_stratified": [int(v) for v in dsm_sample],
        "recon_top_usage": [int(v) for v in rec_top_sample],
        "recon_random_live": [int(v) for v in rec_rand_sample]}

    corr_n = min(n_surv, len(recon_by_rate))
    recon_cols = np.unique(np.concatenate(
        [recon_by_rate[:corr_n], rec_top_sample, rec_rand_sample]))
    Z_dsm, l0_dsm = sweep_collect(models["dsm"], cache["resid"], survivors,
                                  device)
    Z_rec, l0_rec = sweep_collect(models["recon"], cache["resid"], recon_cols,
                                  device)
    out["saturation_check"] = {"dsm_sweep_mean_l0": l0_dsm,
                               "recon_sweep_mean_l0": l0_rec, "k": K}
    print(f"[sweep] mean L0 dsm={l0_dsm:.2f} recon={l0_rec:.2f}", flush=True)
    col_of = {"dsm": {int(l): i for i, l in enumerate(survivors)},
              "recon": {int(l): i for i, l in enumerate(recon_cols)}}
    flush()

    # ---- measurement 1c: pairwise |corr| ---------------------------------
    rows = np.arange(0, Z_dsm.shape[0], corr_row_stride)
    rc = np.array([col_of["recon"][int(l)] for l in recon_by_rate[:corr_n]])
    out["m1c_paircorr"] = {
        "dsm_survivors": {**paircorr_stats(Z_dsm[rows]),
                          "group_mean_rate":
                              float(rates["dsm"][survivors].mean())},
        f"recon_top{corr_n}_by_rate": {
            **paircorr_stats(Z_rec[rows][:, rc]),
            "group_mean_rate":
                float(rates["recon"][recon_by_rate[:corr_n]].mean())},
        "note": "rates differ ~18x between groups; |corr| distributions "
                "carry a rate-driven component (pre-registered caveat)"}
    print(f"[m1c] dsm mean|corr|="
          f"{out['m1c_paircorr']['dsm_survivors']['mean_abs_corr']:.4f} "
          f"recon mean|corr|="
          f"{out['m1c_paircorr'][f'recon_top{corr_n}_by_rate']['mean_abs_corr']:.4f}",
          flush=True)
    flush()

    # ---- measurement 3: max-activating dumps + judge ---------------------
    renderer = TraceRenderer(cap_meta["model"], traces_path, trace_meta)
    valid = valid_window_mask(offsets, Z_dsm.shape[0])
    dump_dir = pathlib.Path(vol) / "backtracking_eval" / "survivor_interp"
    dump_dir.mkdir(parents=True, exist_ok=True)

    def enc_norm_per_offset(kind, lat):
        wcol = models[kind].W_enc[:, int(lat)].detach().reshape(T, dc.D_MODEL)
        return [round(float(v), 3) for v in wcol.norm(dim=1).cpu()]

    groups_spec = [("dsm_survivors", "dsm", Z_dsm, dsm_sample),
                   ("recon_top_usage", "recon", Z_rec, rec_top_sample),
                   ("recon_random_live", "recon", Z_rec, rec_rand_sample)]
    dumps: dict = {}
    for gname, kind, Z, sample in groups_spec:
        g: dict = {}
        for lat in sample:
            vals = Z[:, col_of[kind][int(lat)]]
            sel = top_windows(vals, offsets, valid, n_take=n_top_windows)
            exs = []
            for s, k, a in sel:
                p = s - int(offsets[k])
                exs.append({"win_start_global": s, "trace": k,
                            "qid": trace_meta[k]["qid"], "pos": p,
                            "act": round(a, 3),
                            "ctx": renderer.render(k, p)})
            g[str(int(lat))] = {
                "rate": float(rates[kind][int(lat)]),
                "count": int(counts[kind][int(lat)]),
                "enc_norm_per_offset": enc_norm_per_offset(kind, lat),
                "examples": exs}
        dumps[gname] = g
        (dump_dir / f"dumps_{gname}{tag_suffix}.json").write_text(
            json.dumps(g, indent=1))
        n_full = sum(len(v["examples"]) >= n_top_windows for v in g.values())
        print(f"[m3 dumps] {gname}: {len(g)} latents, {n_full} with "
              f">={n_top_windows} deduped windows", flush=True)
    if commit_cb is not None:
        commit_cb()

    key = os.environ.get("ANTHROPIC_API_KEY") \
        or os.environ.get("ANTHROPIC_API_KEY_MATS")
    out["m3_autointerp"] = {}
    if not (do_judge and key):
        out["m3_autointerp"]["skipped"] = (
            "judge disabled" if not do_judge else "no anthropic key in env; "
            "dumps written to survivor_interp/ for local judging")
        flush()
    else:
        from concurrent.futures import ThreadPoolExecutor

        all_ctx = [(gname, lat, e["ctx"])
                   for gname, g in dumps.items()
                   for lat, v in g.items() for e in v["examples"]]
        for gname, kind, _Z, _sample in groups_spec:
            g = dumps[gname]
            judged, skipped = [], 0
            todo = [(lat, v) for lat, v in g.items()
                    if len(v["examples"]) >= n_top_windows]
            skipped = len(g) - len(todo)

            def one(item, gname=gname):
                lat, v = item
                jr = np.random.default_rng(seed + 7 * int(lat))
                pool = [c for gn, l2, c in all_ctx if l2 != lat]
                decoys = [pool[i] for i in
                          jr.choice(len(pool), 4, replace=False)]
                r = judge_latent(int(lat), v["examples"], decoys, key, jr)
                r["rate"] = v["rate"]
                return r

            with ThreadPoolExecutor(max_workers=6) as pool_ex:
                for r in pool_ex.map(lambda it: _safe(one, it), todo):
                    if r is not None:
                        judged.append(r)
            sc = np.array([r["score"] for r in judged])
            out["m3_autointerp"][gname] = {
                "n_judged": int(len(sc)),
                "n_skipped_lt_windows": int(skipped),
                "score_mean": float(sc.mean()) if len(sc) else None,
                "score_std": float(sc.std()) if len(sc) else None,
                "score_se": float(sc.std() / max(1, len(sc)) ** 0.5)
                if len(sc) else None,
                "score_median": float(np.median(sc)) if len(sc) else None,
                "per_latent": judged}
            print(f"[m3 judge] {gname}: n={len(sc)} "
                  f"mean={sc.mean() if len(sc) else float('nan'):.3f} "
                  f"median={np.median(sc) if len(sc) else float('nan'):.3f}",
                  flush=True)
            flush()

    out["_runtime_s"] = round(time.time() - t0, 1)
    flush()
    print(f"ALL DONE in {out['_runtime_s']}s", flush=True)
    return out


def _safe(fn, item):
    try:
        return fn(item)
    except Exception as e:                                    # noqa: BLE001
        print(f"judge skip: {e}", flush=True)
        return None
