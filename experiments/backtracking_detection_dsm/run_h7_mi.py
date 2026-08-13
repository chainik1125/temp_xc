"""H7 -- l0-matching vs MI-matching on the w6mix_recon preactivations.

Tests the reviewer note's hypothesis H7 (bird_diffusion_sae_txc_proposal.tex
Sec. "Falsifiable hypotheses"): comparisons matched only at expected l0
differ systematically from comparisons additionally matched at estimated
mutual information.

Three activation families applied to the SAME encoder preactivations
a = (x - b_dec) @ W_enc + b_enc of txc_w6_mix/txc_recon/recon_s0.pt
(flat window-6 dict, T*d = 24576 -> H = 16384), on the identical capture /
example-set / probe path as run_detection / run_controls:

  TopK-k        z = scatter(relu(topk(a, k)))            (the trained gate)
  JumpReLU-t    z = a * 1[a > theta]                     (global theta)
  smooth l0     z = relu(a) * sigmoid((a^2-theta^2)/(2*tau) + c), c = 0
                (the note's eq. smooth-jumprelu, 3 temperatures
                tau = beta * theta96^2, beta in {0.05, 0.2, 0.8})

Per setting we measure
  (i)  expected l0 -- nonzero count for TopK/JumpReLU; for the smooth gate
       the posterior support mass E[sum_j q_j 1[a_j > 0]] (the gate q is the
       note's inclusion probability; the raw nonzero count is also reported);
  (ii) the note's Gaussian MI proxy on the CODE as observation:
       1/2 log det(I + Sigma_z / nu0) with FIXED reference noise
       nu0 = (0.1 * rms of nonzero TopK-96 code values)^2, evaluated by the
       Gram trick on an m = 4096 stride-sampled trace-window code sample
       (rank capped at m by construction);
  (iii) downstream sentence-set detection PR-AUC S8 through the unchanged
       detect_core.probe_cv protocol (same folds, labels, Welch-t top-S,
       l1-logistic as every number in this repo).

Matching sets: (A) equal l0 ~= 96 (the repo-wide budget); (B) equal
MI-proxy = MI(TopK-96), reached by bisecting theta per family. All grid
settings are reported so any other matching can be read off the table.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from experiments.backtracking_detection_dsm import detect_core as dc

RECON_REL = "txc_recon/recon_s0.pt"
H_DIM = 16384
K_REF = 96
TOPK_GRID = [24, 48, 96, 192, 384]
JUMP_L0_TARGETS = [24, 48, 96, 192, 384]
SMOOTH_L0_TARGETS = [24, 96, 384]
BETAS = [0.05, 0.2, 0.8]
M_MI = 4096
MI_SEED = 20260813
OUT_NAME = "h7_mi_matching"


def run(device: str = "cuda", vol: str = "/vol",
        traces_path: str = "/work/data/traces.json",
        labels_path: str = "/work/data/sentence_labels.json",
        limit_traces: int | None = None, tag_suffix: str = "",
        commit_cb=None, w6_dir: str = "/vol/txc_w6_mix") -> dict:
    import torch

    t0 = time.time()
    out: dict = {"protocol": {
        "encoder": f"{w6_dir}/{RECON_REL} (w6_flat time-major, T*d=24576, "
                   f"H={H_DIM}); preacts a = (x - b_dec) @ W_enc + b_enc",
        "families": {
            "topk": "z = scatter(relu(topk(a, k)))",
            "jump": "z = a * 1[a > theta], theta global, calibrated on the "
                    "pooled MI-sample preact quantile",
            "smooth": "z = relu(a) * sigmoid((a^2 - theta^2)/(2 tau) + c), "
                      "c = 0; tau = beta * theta96^2, beta in "
                      f"{BETAS}; theta bisected per (tau, target)",
        },
        "l0_def": "topk/jump: mean nonzero count; smooth: mean gate mass "
                  "sum_j q_j 1[a_j>0] (nonzero count reported separately)",
        "mi_proxy": "0.5 * sum log(1 + lambda_i/nu0), lambda_i eigenvalues "
                    f"of Cov(z) on m={M_MI} stride-sampled trace-window "
                    "codes via the m x m Gram trick (rank <= m); nu0 = "
                    "(0.1 * rms nonzero TopK-96 code value)^2, fixed "
                    "across every setting",
        "probe": "detect_core.probe_cv unchanged: l1-logistic C=1 "
                 "liblinear, 5-fold GroupKFold by trace, top-S Welch-t, "
                 "S in {8,32}; sentence example set (paper D+/D- split)",
        "matchings": {"A_equal_l0": "l0 ~= 96",
                      "B_equal_mi": "MI-proxy = MI(TopK-96), theta "
                                    "bisected per family"},
        "mi_sample_seed": MI_SEED,
    }}

    def flush():
        d = pathlib.Path(vol) / "backtracking_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{OUT_NAME}{tag_suffix}.json").write_text(
            json.dumps(out, indent=1, default=str))
        if commit_cb is not None:
            commit_cb()

    # ---- capture + example sets (identical to run_controls) ---------------
    cache, offsets, trace_meta, cap_meta = dc.capture_traces(
        traces_path, labels_path, device=device, limit_traces=limit_traces)
    out["capture"] = cap_meta
    out["capture"]["n_traces"] = len(trace_meta)
    ex = dc.collect_examples(trace_meta)
    examples = ex["sentence"]
    y = np.array([e[2] for e in examples])
    groups = np.array([e[0] for e in examples])
    out["example_set"] = {"name": "sentence", "n": int(len(y)),
                          "n_pos": int(y.sum()),
                          "positive_base_rate": float(y.mean()),
                          "n_groups": int(len(set(groups.tolist())))}
    print(f"[examples/sentence] n={len(y)} pos={int(y.sum())}", flush=True)
    W_sent = dc.gather_windows(cache["resid"], offsets, examples)

    model, meta = dc.load_w6_dict(str(pathlib.Path(w6_dir) / RECON_REL),
                                  "recon", device)
    out["dict_meta"] = meta
    chk = dc.w6_nmse_both_orders(model, W_sent, device)
    out["w6_ordering_check"] = chk
    print(f"[w6 ordering] NMSE time_major={chk['nmse_time_major']:.4f} "
          f"dim_major={chk['nmse_dim_major']:.4f}", flush=True)
    if not chk["nmse_time_major"] < 0.9 * chk["nmse_dim_major"]:
        flush()
        raise RuntimeError("w6 flatten-order gate failed; not scoring")

    # ---- preactivations ---------------------------------------------------
    @torch.no_grad()
    def preacts_of_windows(Wt):
        ps = []
        for s in range(0, Wt.shape[0], 512):
            x = Wt[s:s + 512].to(device, torch.float32)
            ps.append(model.encode_pre(x.reshape(x.shape[0], -1)))
        return torch.cat(ps)

    pre_sent = preacts_of_windows(W_sent)                      # GPU fp32
    T = len(dc.WIN_OFF)
    n_win = cache["resid"].shape[0] - T + 1
    rng = np.random.default_rng(MI_SEED)
    starts = np.sort(rng.choice(n_win, size=min(M_MI, n_win), replace=False))
    mi_parts = []
    for s0 in range(0, len(starts), 512):
        blk = starts[s0:s0 + 512]
        Xb = np.stack([np.asarray(cache["resid"][s:s + T]) for s in blk])
        mi_parts.append(torch.from_numpy(Xb.astype(np.float32)))
    W_mi = torch.cat(mi_parts)                                 # (m, T, d)
    pre_mi = preacts_of_windows(W_mi)
    del cache, mi_parts
    m = pre_mi.shape[0]
    out["protocol"]["m_mi_realized"] = int(m)

    # ---- family activations ----------------------------------------------
    def z_of(pre, family, k=None, theta=None, tau=None):
        if family == "topk":
            v, i = pre.topk(k, dim=1)
            return torch.zeros_like(pre).scatter(1, i, torch.relu(v))
        if family == "jump":
            return pre * (pre > theta)
        if family == "smooth":
            q = torch.sigmoid((pre * pre - theta * theta) / (2.0 * tau))
            return torch.relu(pre) * q
        raise ValueError(family)

    def l0_gate(pre, family, k=None, theta=None, tau=None):
        if family == "smooth":
            q = torch.sigmoid((pre * pre - theta * theta) / (2.0 * tau))
            return float((q * (pre > 0)).sum(1).mean())
        z = z_of(pre, family, k=k, theta=theta, tau=tau)
        return float((z > 0).float().sum(1).mean())

    nu0_holder = {}

    @torch.no_grad()
    def mi_proxy(z):
        zc = z - z.mean(0, keepdim=True)
        Gm = (zc @ zc.T).double() / z.shape[0]
        lam = torch.linalg.eigvalsh(Gm).clamp_min(0)
        return float(0.5 * torch.log1p(lam / nu0_holder["nu0"]).sum())

    z96_mi = z_of(pre_mi, "topk", k=K_REF)
    nz = z96_mi[z96_mi > 0]
    s96 = float(nz.pow(2).mean().sqrt())
    nu0_holder["nu0"] = (0.1 * s96) ** 2
    out["protocol"]["nu0"] = nu0_holder["nu0"]
    out["protocol"]["rms_nonzero_topk96_code"] = s96
    mi_ref = mi_proxy(z96_mi)
    out["protocol"]["mi_target_topk96_nats"] = mi_ref
    print(f"[nu0] s96={s96:.4f} nu0={nu0_holder['nu0']:.6f} "
          f"MI(topk96)={mi_ref:.2f} nats", flush=True)
    del z96_mi, nz

    # pooled preact quantile thresholds for JumpReLU targets
    flat = pre_mi.flatten()
    flat_sorted = torch.sort(flat).values
    N = flat_sorted.shape[0]

    def theta_for_count(t):
        idx = int(round(N * (1.0 - t / H_DIM)))
        th = float(flat_sorted[min(max(idx, 0), N - 1)])
        return max(th, 1e-6)

    theta_jump = {t: theta_for_count(t) for t in JUMP_L0_TARGETS}
    theta96 = theta_jump[96]
    taus = {b: b * theta96 ** 2 for b in BETAS}
    out["protocol"]["theta_jump_calibrated"] = theta_jump
    out["protocol"]["theta96"] = theta96
    out["protocol"]["taus"] = {str(b): taus[b] for b in BETAS}
    frac_pos = float((flat > 0).float().mean())
    out["protocol"]["frac_positive_preacts"] = frac_pos
    print(f"[cal] theta96={theta96:.4f} frac_pos={frac_pos:.4f} "
          f"taus={ {b: round(taus[b], 4) for b in BETAS} }", flush=True)
    del flat, flat_sorted

    def bisect(fn, target, lo, hi, iters=40):
        """fn monotone decreasing in theta; find fn(theta) = target."""
        flo, fhi = fn(lo), fn(hi)
        if not (flo >= target >= fhi):
            return None, {"lo": lo, "hi": hi, "f_lo": flo, "f_hi": fhi}
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if fn(mid) >= target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), None

    # ---- score one arm ----------------------------------------------------
    out["arms"] = {}

    def score(tag, family, **kw):
        z_mi = z_of(pre_mi, family, **kw)
        mi = mi_proxy(z_mi)
        l0_mi = l0_gate(pre_mi, family, **kw)
        nnz_mi = float((z_mi > 0).float().sum(1).mean())
        del z_mi
        z_s = z_of(pre_sent, family, **kw)
        l0_s = l0_gate(pre_sent, family, **kw)
        nnz_s = float((z_s > 0).float().sum(1).mean())
        F = z_s.abs().cpu().numpy().astype(np.float32)
        del z_s
        res = dc.probe_cv(F, y, groups)
        del F
        arm = {"family": family,
               **{k2: (float(v2) if v2 is not None else None)
                  for k2, v2 in kw.items()},
               "l0_gate_mi_sample": l0_mi, "l0_nonzero_mi_sample": nnz_mi,
               "l0_gate_sentence": l0_s, "l0_nonzero_sentence": nnz_s,
               "mi_proxy_nats": mi, **res}
        out["arms"][tag] = arm
        print(f"[arm {tag}] l0={l0_mi:.1f} (nnz {nnz_mi:.0f}) "
              f"MI={mi:.1f} | S8 PR={res['prauc_S8']:.4f} "
              f"S32 PR={res['prauc_S32']:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        flush()
        return arm

    # sweep: topk
    for k in TOPK_GRID:
        score(f"topk_k{k}", "topk", k=k)
    # sweep: jumprelu at calibrated l0 targets
    for t in JUMP_L0_TARGETS:
        score(f"jump_l{t}", "jump", theta=theta_jump[t])
    # sweep: smooth gate, theta bisected per (tau, target) on gate mass
    for b in BETAS:
        tau = taus[b]
        for t in SMOOTH_L0_TARGETS:
            fn = lambda th: l0_gate(pre_mi, "smooth", theta=th, tau=tau)  # noqa: E731
            th, err = bisect(fn, float(t), 1e-6, 12.0 * theta96)
            if th is None:
                out["arms"][f"smooth_b{b}_l{t}"] = {
                    "family": "smooth", "beta": b, "tau": tau,
                    "calibration_failed": err}
                flush()
                continue
            score(f"smooth_b{b}_l{t}", "smooth", theta=th, tau=tau)

    # MI-matched settings (target = MI(topk96)). MI is monotone decreasing
    # in theta, so theta -> 0 gives the family's maximum reachable MI at
    # this nu0; if even that is below the target the match is genuinely
    # unreachable and the closest-achievable setting is scored with an
    # explicit flag (an informative H7 outcome in itself, not an error).
    def mi_match(tag, family, fn, **kw):
        th, err = bisect(fn, mi_ref, 1e-6, theta96 * 16.0, iters=16)
        if th is not None:
            score(tag, family, theta=th, **kw)
            return
        mi_at_min = fn(1e-6)
        arm = score(tag, family, theta=1e-6, **kw)
        arm["mi_unreachable"] = True
        arm["mi_max_reachable_nats"] = mi_at_min
        arm["mi_shortfall_nats"] = mi_ref - mi_at_min
        arm["bracket_diagnostics"] = err
        flush()

    fnj = lambda th: mi_proxy(z_of(pre_mi, "jump", theta=th))  # noqa: E731
    mi_match("jump_miMatch", "jump", fnj)
    for b in BETAS:
        tau = taus[b]
        fns = lambda th, tau=tau: mi_proxy(  # noqa: E731
            z_of(pre_mi, "smooth", theta=th, tau=tau))
        mi_match(f"smooth_b{b}_miMatch", "smooth", fns, tau=tau)

    # ---- matched comparison tables ---------------------------------------
    def row(tag):
        a = out["arms"].get(tag, {})
        r = {"tag": tag,
             "l0": a.get("l0_gate_mi_sample"),
             "mi": a.get("mi_proxy_nats"),
             "prauc_S8": a.get("prauc_S8"),
             "prauc_S32": a.get("prauc_S32")}
        if a.get("mi_unreachable"):
            r["mi_unreachable"] = True
            r["mi_shortfall_nats"] = a.get("mi_shortfall_nats")
        return r

    out["matched"] = {
        "A_equal_l0_96": [row("topk_k96"), row("jump_l96")] +
                         [row(f"smooth_b{b}_l96") for b in BETAS],
        "B_equal_mi_topk96": [row("topk_k96"), row("jump_miMatch")] +
                             [row(f"smooth_b{b}_miMatch") for b in BETAS],
    }
    out["_runtime_s"] = round(time.time() - t0, 1)
    flush()
    print(f"H7 ALL DONE in {out['_runtime_s']}s", flush=True)
    return out
